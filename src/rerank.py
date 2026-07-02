"""
rerank.py — optional cross-encoder rerank pass for the DuckBot brain.

After hybrid retrieval (vector + BM25 + RRF) returns top-N*3 candidates,
we can optionally rerank them with a cross-encoder for a substantial recall
boost. This is the single biggest recall improvement we can add at our scale
(see RESEARCH.md "Layer 7 candidate").

Sources (verified via GitHub REST API 2026-06-23):
  - qwen3-reranker-0.6b — local reranker default
    https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
  - huggingface/sentence-transformers (CrossEncoder API) — Apache-2.0
    https://github.com/huggingface/sentence-transformers

We re-implement the integration pattern rather than copy code, so the
LICENSE stays clean. The pattern itself is from the sentence-transformers
README + the mem0 SentenceTransformerReranker bug history (issue #4033):
cross-encoder models must use `CrossEncoder`, not `SentenceTransformer`,
or scores silently collapse to 0.0.

Design:
  - Lazy model load — only loads when `rerank()` is first called.
  - Failure-safe — if anything throws, returns the input list unchanged.
  - Zero paid APIs. Pure local inference.
  - Three backends, auto-detected in priority order:
      1. sentence-transformers CrossEncoder (if `sentence-transformers` installed)
      2. LM Studio rerank endpoint (if `LMSTUDIO_RERANK_URL` env set)
      3. No-op (returns input order unchanged)

Activation: opt-in via env var `DUCKBOT_RERANK=1` or per-call argument.
Default OFF — keeps current RRF behavior identical for callers that
don't ask for it.

Cost: the Qwen3 reranker default is still local and modest-sized; on an
M-series Mac it typically lands in the same sub-200ms/query bucket as
the vector + BM25 hot path, depending on batch size and hardware.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# Default model for sentence-transformers / Hugging Face downloads.
# Keep this as the canonical repo id so the CrossEncoder path can load it.
DEFAULT_RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"

# Default model for LM Studio's local rerank endpoint. LM Studio uses the
# model id as exposed by the local server, which in this repo is the
# lowercase alias the user specified.
DEFAULT_LMSTUDIO_RERANK_MODEL = "qwen3-reranker-0.6b"

# Truncate documents to this many chars before scoring. Cross-encoders
# are token-bounded. Roughly 1500 chars ≈ 400 tokens with our typical
# chunk text, leaving headroom for the default Qwen3 reranker.
MAX_DOC_CHARS = 1500

# Max query/doc pair batch size for predict(). 32 fits comfortably in
# memory for the default local reranker on typical developer hardware.
DEFAULT_BATCH_SIZE = 32


# -----------------------------------------------------------------------------
# Result type
# -----------------------------------------------------------------------------


@dataclass
class RerankResult:
    """One reranked hit."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    tier: str = "unknown"
    original_score: float = 0.0  # RRF score from the retriever
    rerank_score: float = 0.0  # cross-encoder relevance (higher = better)
    final_score: float = 0.0  # 0.7*rerank + 0.3*original (or whatever weight)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": self.metadata,
            "tier": self.tier,
            "original_score": self.original_score,
            "rerank_score": self.rerank_score,
            "final_score": self.final_score,
        }


# -----------------------------------------------------------------------------
# Backend protocol
# -----------------------------------------------------------------------------


class RerankBackend(Protocol):
    """Anything that can score (query, doc) pairs."""

    name: str

    def score(self, query: str, docs: list[str]) -> list[float]:
        """Return a relevance score for each doc. Higher = more relevant."""
        ...


class NoopBackend:
    """Returns the input order unchanged with constant scores. Used when
    no rerank backend is available — gives the caller a stable API."""

    name = "noop"

    def score(self, query: str, docs: list[str]) -> list[float]:
        # Mild boost for shorter docs (often more on-point for narrow queries).
        # This is a fallback heuristic, not a real cross-encoder.
        return [1.0 / (1 + i * 0.1) for i in range(len(docs))]


class SentenceTransformersBackend:
    """Local cross-encoder via `sentence-transformers` (Apache-2.0)."""

    def __init__(self, model_name: str | None = None):
        # Import inside __init__ so the module is optional.
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            ) from e
        # NB: must be CrossEncoder, NOT SentenceTransformer. mem0 hit this
        # in issue #4033 — cross-encoder models silently fail with mean pooling.
        model_name = model_name or os.environ.get("DUCKBOT_RERANK_MODEL", DEFAULT_RERANK_MODEL)
        self.model = CrossEncoder(model_name, max_length=512)
        self.name = f"cross-encoder:{model_name}"

    def score(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        # Truncate docs to the model's expected length.
        truncated = [(d or "")[:MAX_DOC_CHARS] for d in docs]
        pairs = [(query, d) for d in truncated]
        # predict() returns a numpy array of floats.
        raw = self.model.predict(pairs, batch_size=DEFAULT_BATCH_SIZE, show_progress_bar=False)
        # Cross-encoder rerankers often output logits (can be negative).
        # Sigmoid to 0..1 for a stable score scale.
        try:
            import math

            scores = [1.0 / (1.0 + math.exp(-float(s))) for s in raw]
        except Exception:
            scores = [float(s) for s in raw]
        return scores


class Qwen3RerankerBackend:
    """Rerank via /v1/chat/completions using qwen3-0.6b-reranker.

    LM Studio serves it as a chat model, NOT a /rerank endpoint.
    We format docs as a prompt and parse float scores from the completion.
    """

    def __init__(self, url=None, model=None):
        import os as _o
        self.url = url or _o.environ.get("LMSTUDIO_RERANK_URL",
                  "http://127.0.0.1:1234/v1/chat/completions")
        self.model = model or _o.environ.get("LMSTUDIO_RERANK_MODEL",
                     "qwen3-0.6b-reranker")
        self.name = "qwen3-reranker:" + self.model

    def score(self, query, docs):
        if not docs:
            return []
        import httpx, os as _o, re
        trunc = [(d or "")[:1500] for d in docs]
        numbered = "\n".join("{}. {}".format(i + 1, d) for i, d in enumerate(trunc))
        prompt = (
            "Query: " + query + "\n\n"
            "Rate each document relevance from 0.0 to 1.0.\n"
            'Output ONLY valid JSON: {"scores": [0.1, 0.9, 0.3]}\n\n'
            + numbered
        )
        key = _o.environ.get("LMSTUDIO_KEY", _o.environ.get("LMSTUDIO_API_KEY", ""))
        hdrs = {"Content-Type": "application/json"}
        if key:
            hdrs["Authorization"] = "Bearer " + key
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a relevance scorer. Output ONLY valid JSON."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 300,
        }
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(self.url, json=payload, headers=hdrs)
            resp.raise_for_status()
            text = (resp.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", ""))
            found = re.findall(r"[-+]?\d*\.?\d+", text)
            scores = [float(x) for x in found[:len(docs)]]
            if len(scores) == len(docs):
                return scores
            return [0.0] * len(docs)
        except Exception:
            return [0.0] * len(docs)

    def available(self):
        # Actually check LM Studio — don't assume just because the model is loaded
        try:
            import httpx
            with httpx.Client(timeout=5.0) as client:
                resp = client.get("http://127.0.0.1:1234/v1/models")
            return resp.status_code == 200
        except Exception:
            return False


class LMStudioBackend:
    """LM Studio rerank endpoint. Some LM Studio builds expose a
    `/v1/rerank` route (e.g. via the llm-rerank plugin or TEI bridge)."""

    def __init__(self, url: str | None = None, model: str | None = None):
        self.url = url or os.environ.get(
            "LMSTUDIO_RERANK_URL", "http://127.0.0.1:1234/v1/rerank"
        )
        self.model = model or os.environ.get(
            "LMSTUDIO_RERANK_MODEL", DEFAULT_LMSTUDIO_RERANK_MODEL
        )
        self.name = f"lmstudio:{self.url}"

    async def _score_async(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        import httpx  # already a dep
        truncated = [(d or "")[:MAX_DOC_CHARS] for d in docs]
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    self.url,
                    json={"model": self.model, "query": query, "documents": truncated},
                )
            resp.raise_for_status()
            data = resp.json()
            return _parse_cohere_rerank_response(data, len(docs))
        except Exception as e:
            logger.warning("LM Studio rerank failed: %s — falling back to input order", e)
            return [0.0] * len(docs)

    def available(self) -> bool:
        """Return True if the rerank endpoint responds with a non-error."""
        try:
            import httpx
            import os as _os
            key = (
                _os.environ.get("LMSTUDIO_KEY")
                or _os.environ.get("LMSTUDIO_API_KEY")
                or _os.environ.get("LM_API_TOKEN")
                or ""
            )
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            with httpx.Client(timeout=5.0) as client:
                resp = client.post(
                    self.url,
                    json={"model": self.model, "query": "test", "documents": ["doc"]},
                    headers=headers,
                )
                # Any non-404/401/500 class error means the endpoint exists
                return resp.status_code < 500 and "error" not in resp.json()
        except Exception:
            return False

    def score(self, query: str, docs: list[str]) -> list[float]:
        # sync wrapper for the Protocol interface
        import concurrent.futures
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = None
        if loop is not None and loop.is_running():
            # We're inside an event loop already. asyncio.run() would raise
            # RuntimeError ("cannot be called from a running event loop") and
            # loop.run_until_complete() would too. Run the coroutine on a
            # worker thread with its own loop instead.
            def _runner():
                return asyncio.run(self._score_async(query, docs))
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                return ex.submit(_runner).result()
        if loop is not None:
            return loop.run_until_complete(self._score_async(query, docs))
        return asyncio.run(self._score_async(query, docs))


def _parse_cohere_rerank_response(data: dict, n_docs: int) -> list[float]:
    """Parse a Cohere-compatible /v1/rerank response into per-document scores.

    LM Studio's /v1/rerank is Cohere-compatible. Results are sorted by
    relevance_score (highest first), NOT in original document order. Each
    result has an `index` field pointing back to the input documents array.
    We map by index so the caller gets scores aligned with their input.
    """
    results = data.get("results") or []
    scores = [0.0] * n_docs
    for r in results:
        idx = int(r.get("index", -1))
        if 0 <= idx < n_docs:
            scores[idx] = float(r.get("relevance_score", r.get("score", 0.0)))
    return scores


# -----------------------------------------------------------------------------
# Lazy backend resolution
# -----------------------------------------------------------------------------


_BACKEND: RerankBackend | None = None
_BACKEND_TRIED: set[str] = set()


def _resolve_backend(prefer: str | None = None) -> RerankBackend:
    """Pick the best available backend.

    Priority:
      1. LM Studio rerank endpoint (local, no network)
      2. sentence-transformers CrossEncoder (if `sentence-transformers`
         is installed and a Hugging Face model is available)
      3. noop

    The choice is cached after the first successful init so subsequent
    calls are O(1).
    """
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    def _cache_if_real(backend: RerankBackend) -> RerankBackend:
        if isinstance(backend, NoopBackend) or getattr(backend, "name", None) == "noop":
            return backend
        global _BACKEND
        _BACKEND = backend
        return backend

    # Priority 1: qwen3-0.6b-reranker via /v1/chat/completions (the loaded model)
    if prefer in (None, "lmstudio", "auto"):
        try:
            be = Qwen3RerankerBackend()
            if be.available():
                _cache_if_real(be)
                logger.info("rerank backend: %s", be.name)
                return be
        except Exception as e:
            logger.debug("Qwen3 reranker backend unavailable: %s", e)

    # Priority 2: Cohere-compatible /v1/rerank (LM Studio builds with the plugin)
    if prefer in (None, "lmstudio", "auto"):
        try:
            be = LMStudioBackend()
            if be.available():
                _cache_if_real(be)
                logger.info("rerank backend: %s", be.name)
                return be
            else:
                logger.debug("LM Studio /rerank endpoint not available; skipping")
        except Exception as e:
            logger.debug("LM Studio rerank backend unavailable: %s", e)

    if prefer in (None, "sentence-transformers", "auto"):
        try:
            be = SentenceTransformersBackend()
            _cache_if_real(be)
            logger.info("rerank backend: %s", be.name)
            return be
        except Exception as e:
            logger.debug("sentence-transformers backend unavailable: %s", e)

    logger.info("rerank backend: noop (no local model; pass-through)")
    return NoopBackend()


def reset_backend() -> None:
    """Force re-resolution on next call. Used by tests."""
    global _BACKEND
    _BACKEND = None


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def rerank_available() -> bool:
    """True if a real cross-encoder backend can be loaded. False if noop."""
    be = _resolve_backend()
    return not isinstance(be, NoopBackend)


def rerank(
    query: str,
    candidates: list[RerankResult] | list[dict[str, Any]],
    *,
    backend: RerankBackend | None = None,
    weight_original: float = 0.3,
    weight_rerank: float = 0.7,
    top_k: int | None = None,
) -> list[RerankResult]:
    """Rerank a list of hybrid-retrieved candidates.

    Args:
        query: The original query string.
        candidates: Either RerankResult objects OR plain dicts with at
            minimum `id`, `text`, and ideally `tier` + `metadata`. Plain
            dicts are promoted to RerankResult internally.
        backend: Optional explicit backend (skips auto-detect). Used in tests.
        weight_original: How much the original RRF score contributes to
            final_score. Default 0.3.
        weight_rerank: How much the cross-encoder score contributes.
            Default 0.7 (cross-encoder is more accurate at our scale).
        top_k: If set, truncate to this many results after rerank.
            Otherwise return all input candidates in reranked order.

    Returns:
        Sorted list of RerankResult, highest final_score first.

    Failure mode: if the backend throws, the input list is returned
    unchanged (with original_score preserved). The caller never gets an
    exception from this function — rerank is best-effort.
    """
    if not candidates:
        return []

    started = time.time()

    # Normalize input to RerankResult objects.
    norm: list[RerankResult] = []
    for c in candidates:
        if isinstance(c, RerankResult):
            norm.append(c)
        else:
            norm.append(
                RerankResult(
                    id=str(c.get("id") or c.get("chunk_id") or ""),
                    text=str(c.get("text", "")),
                    metadata=dict(c.get("metadata") or {}),
                    tier=str(c.get("tier", "unknown")),
                    original_score=float(c.get("original_score") or c.get("rrf_score") or 0.0),
                )
            )

    # Pull the original RRF score before we overwrite it.
    for r in norm:
        r.final_score = r.original_score

    be = backend or _resolve_backend()

    # Noop → preserve original ordering; original_score is already final_score.
    if isinstance(be, NoopBackend):
        if top_k is not None:
            norm = norm[:top_k]
        logger.debug("rerank: noop backend, returning %d candidates unchanged", len(norm))
        return norm

    docs = [r.text for r in norm]
    try:
        scores = be.score(query, docs)
    except Exception as e:
        logger.warning(
            "rerank backend %s failed (%s); returning input order unchanged",
            be.name,
            e,
        )
        if top_k is not None:
            norm = norm[:top_k]
        return norm

    if len(scores) != len(norm):
        logger.warning(
            "rerank backend returned %d scores for %d candidates; using input order",
            len(scores),
            len(norm),
        )
        if top_k is not None:
            norm = norm[:top_k]
        return norm

    # Combine: final = weight_rerank * rerank + weight_original * original_normalized.
    # The original RRF scores are tiny (1/(60+rank) for k=60), so we min-max
    # normalize them to [0, 1] before mixing, otherwise the rerank term
    # would dominate everything just because it's on a 0..1 scale.
    orig_scores = [r.original_score for r in norm]
    if orig_scores:
        lo, hi = min(orig_scores), max(orig_scores)
        span = hi - lo if hi > lo else 1.0
    else:
        lo, hi, span = 0.0, 0.0, 1.0

    for r, rs in zip(norm, scores):
        r.rerank_score = float(rs)
        norm_orig = (r.original_score - lo) / span if span else 0.0
        r.final_score = weight_rerank * r.rerank_score + weight_original * norm_orig

    norm.sort(key=lambda r: r.final_score, reverse=True)
    if top_k is not None:
        norm = norm[:top_k]

    logger.debug(
        "rerank: %s scored %d candidates in %.0fms (top score=%.3f)",
        be.name,
        len(norm),
        (time.time() - started) * 1000,
        norm[0].final_score if norm else 0.0,
    )
    return norm


# -----------------------------------------------------------------------------
# Convenience: post-RRF hook for src.query.hybrid_query
# -----------------------------------------------------------------------------


def maybe_rerank(
    query: str,
    results: list[Any],  # list of QueryResult — kept duck-typed to avoid import cycle
    *,
    enabled: bool | None = None,
    top_k: int | None = None,
) -> list[Any]:
    """Drop-in rerank step for `hybrid_query` output.

    Args:
        query: The original query.
        results: List of QueryResult from `hybrid_query`.
        enabled: If True, rerank. If False, return input unchanged.
            If None, read DUCKBOT_RERANK env var.
        top_k: If set, truncate to top_k after rerank. If None, return
            all reranked results.

    Returns:
        Re-ordered list of the same QueryResult objects (mutated in place
        for rrf_score; new field `rerank_score` may be added via attribute).

    This is the integration point used by src/query.py.
    """
    if enabled is None:
        enabled = os.environ.get("DUCKBOT_RERANK", "0").lower() in ("1", "true", "yes")

    if not enabled or not results:
        return results

    # Project QueryResult → RerankResult, then map back.
    projected: list[RerankResult] = []
    for r in results:
        projected.append(
            RerankResult(
                id=r.chunk_id,
                text=r.text,
                metadata=dict(r.metadata or {}),
                tier=r.tier,
                original_score=float(r.rrf_score),
            )
        )

    reranked = rerank(query, projected, top_k=top_k)

    # Map back: update original QueryResult.rrf_score with final_score
    # so downstream sorting and formatting still work.
    id_to_final = {r.id: r.final_score for r in reranked}
    id_to_meta = {r.id: r.rerank_score for r in reranked}
    for r in results:
        if r.chunk_id in id_to_final:
            r.rrf_score = id_to_final[r.chunk_id]
        # Stash the raw rerank score in metadata for debugging.
        if r.chunk_id in id_to_meta:
            r.metadata = dict(r.metadata or {})
            r.metadata["rerank_score"] = id_to_meta[r.chunk_id]

    # Re-sort by the new RRF (now: final_score) value.
    results.sort(key=lambda r: r.rrf_score, reverse=True)
    if top_k is not None:
        results = results[:top_k]
    return results


__all__ = [
    "RerankResult",
    "RerankBackend",
    "NoopBackend",
    "SentenceTransformersBackend",
    "LMStudioBackend",
    "rerank",
    "maybe_rerank",
    "rerank_available",
    "reset_backend",
    "DEFAULT_RERANK_MODEL",
    "DEFAULT_LMSTUDIO_RERANK_MODEL",
    "_parse_cohere_rerank_response",
]
