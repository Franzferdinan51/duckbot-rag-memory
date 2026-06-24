"""
embeddings.py — embedding model wrapper.

Three providers, switchable via DUCKBOT_EMBEDDING env var:

  openai   — OpenAI text-embedding-3-small (default; 1536d, $0.02/1M tokens)
             requires OPENAI_API_KEY
  local    — sentence-transformers bge-small-en-v1.5 (free, slower, 384d)
             requires: pip install sentence-transformers
  lmstudio — any LM Studio OpenAI-compatible server
             defaults to http://127.0.0.1:1234/v1
             dim is detected from /v1/models or set via LMSTUDIO_EMBED_DIM
             (typically 384 for bge-small or 1024 for bge-large)

  minimax  — MiniMax embeddings API (paid, high quality)
             requires MINIMAX_API_KEY, defaults to https://api.minimax.io/v1
             uses text-embedding-01 model, 1536d

LM Studio is preferred for self-hosted / privacy-first operation.
The fallback chain is: DUCKBOT_EMBEDDING env > auto-detect LM Studio > OpenAI.
"""

from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Protocol

import httpx


# ---------------------------------------------------------------------------
# .env loader — runs at import time so any entry point gets a populated env.
# Idempotent, silent, and doesn't override already-set vars.
# ---------------------------------------------------------------------------
def _load_dotenv() -> None:
    # Use pathlib (cross-platform: handles Win/Mac/Linux separators).
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        return
    try:
        from dotenv import load_dotenv  # optional dep
        load_dotenv(str(env_file), override=False)
        return
    except ImportError:
        pass
    # Fallback: manual parse
    try:
        with open(env_file, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


_load_dotenv()


class EmbeddingProvider(Protocol):
    """Pluggable embedding interface. All providers must return float32 lists."""

    name: str
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_one(self, text: str) -> list[float]: ...


class EmbeddingError(RuntimeError):
    """Raised when an embedding provider cannot produce a result."""


@dataclass
class OpenAIEmbeddings:
    """OpenAI text-embedding-3-small (or -large) provider."""

    model: str = "text-embedding-3-small"
    name: str = "openai"
    dim: int = 1536
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    batch_size: int = 100  # OpenAI allows up to 2048 inputs per request

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise EmbeddingError(
                "OpenAI API key not set. Set OPENAI_API_KEY env var or pass api_key=..."
            )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            payload = {"model": self.model, "input": batch}
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    f"{self.base_url}/embeddings",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
            vectors = [item["embedding"] for item in data["data"]]
            results.extend(vectors)
        return results

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]


@dataclass
class LMStudioEmbeddings:
    """LM Studio OpenAI-compatible /v1/embeddings.

    LM Studio is a local model server. It speaks the OpenAI API but with a
    dynamic model list — we discover the loaded embedding model from
    /v1/models and use whatever's there.

    Configure via:
      LMSTUDIO_URL    — default http://127.0.0.1:1234
      LMSTUDIO_MODEL  — model id (default: auto-detect first embedding model)
      LMSTUDIO_EMBED_DIM — dim (default: 1024 for bge-large, 384 for bge-small)
                           The actual dim is set by the model; we use this as
                           a hint and trust the server's response.
      LMSTUDIO_API_KEY / LMSTUDIO_KEY / LM_API_TOKEN
                        — Bearer token. LM Studio's recent builds require auth.
                          If unset, falls back to "lm-studio" (the previous
                          no-auth placeholder).
    """

    base_url: str = ""
    model: str = ""
    name: str = "lmstudio"
    dim: int = 1024  # sensible default; will be updated by auto-detect
    api_key: str = "lm-studio"  # LM Studio ignores auth
    batch_size: int = 32  # smaller batches; LM Studio runs on consumer GPUs

    def __post_init__(self) -> None:
        if not self.base_url:
            self.base_url = os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234/v1")
        if not self.model:
            self.model = os.environ.get("LMSTUDIO_MODEL", "text-embedding-embeddinggemma-300m")
        if not self.api_key or self.api_key == "lm-studio":
            # Try common env var names
            self.api_key = (
                os.environ.get("LMSTUDIO_API_KEY")
                or os.environ.get("LMSTUDIO_KEY")
                or os.environ.get("LM_API_TOKEN")
                or "lm-studio"
            )
        if "LMSTUDIO_EMBED_DIM" in os.environ:
            try:
                self.dim = int(os.environ["LMSTUDIO_EMBED_DIM"])
            except ValueError:
                pass

    async def _resolve_dim(self) -> None:
        """Try to learn the actual embedding dim from a test query."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.post(
                    f"{self.base_url}/embeddings",
                    json={"model": self.model, "input": ["test"]},
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    vec = data["data"][0]["embedding"]
                    self.dim = len(vec)
        except Exception:
            pass  # fall back to default

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # Lazy dim resolution on first embed
        results: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            payload = {"model": self.model, "input": batch}
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self.base_url}/embeddings",
                    json=payload,
                    headers=headers,
                )
                resp.raise_for_status()
                data = resp.json()
            vectors = [item["embedding"] for item in data["data"]]
            if vectors and self.dim != len(vectors[0]):
                self.dim = len(vectors[0])
            results.extend(vectors)
        return results

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]


@dataclass
class LocalEmbeddings:
    """Local sentence-transformers provider. Used for offline / cost-free mode."""

    model_name: str = "BAAI/bge-small-en-v1.5"
    name: str = "local"
    dim: int = 384
    _model: object = None  # lazy-loaded SentenceTransformer

    def __post_init__(self) -> None:
        try:
            import sentence_transformers  # noqa: F401
        except ImportError as exc:
            raise EmbeddingError(
                "sentence-transformers not installed. pip install sentence-transformers"
            ) from exc

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        import asyncio
        loop = asyncio.get_event_loop()
        vectors = await loop.run_in_executor(
            None, lambda: model.encode(texts, normalize_embeddings=True).tolist()
        )
        if vectors and self.dim != len(vectors[0]):
            self.dim = len(vectors[0])
        return vectors

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]


@dataclass
class MiniMaxEmbeddings:
    """MiniMax Embeddings API provider.

    Uses the minimax-portal MiniMax embeddings endpoint. Response shape is
    OpenAI-incompatible: returns {"vectors": [[...]], "base_resp": {...}}
    instead of {"data": [{"embedding": [...]}]}.

    Required request body uses:
      - "texts" (list[str])  not "input"
      - "type"  ("db" or "query" — db for indexing, query for retrieval)

    Configure via:
      MINIMAX_API_KEY     — required
      MINIMAX_BASE_URL    — default https://api.minimax.io/v1
      MINIMAX_EMBED_MODEL — default text-embedding-01 (1536d)
      MINIMAX_EMBED_TYPE  — default "db" (for ingest); switch to "query" for retrieval
                            (different optimization pass; matters for accuracy)
    """

    model: str = "text-embedding-01"
    name: str = "minimax"
    dim: int = 1536
    api_key: str = ""
    base_url: str = ""
    batch_size: int = 32  # RPM limit is tight; small batches
    embed_type: str = ""  # "db" for indexing, "query" for retrieval
    max_retries: int = 3
    retry_base_delay: float = 5.0  # seconds, for rate-limit backoff

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("MINIMAX_API_KEY", "")
        if not self.api_key:
            raise EmbeddingError(
                "MiniMax API key not set. Set MINIMAX_API_KEY env var."
            )
        if not self.base_url:
            self.base_url = os.environ.get("MINIMAX_BASE_URL", "https://api.minimax.io/v1")
        if "MINIMAX_EMBED_MODEL" in os.environ:
            self.model = os.environ["MINIMAX_EMBED_MODEL"]
        if "MINIMAX_EMBED_TYPE" in os.environ:
            self.embed_type = os.environ["MINIMAX_EMBED_TYPE"]
        if not self.embed_type:
            self.embed_type = "db"  # default for ingest

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        """One HTTP call. Raises EmbeddingError on rate-limit or HTTP error.

        MiniMax returns HTTP 200 even on rate-limit — the actual error is in
        base_resp.status_code. We treat 1002/1003/1004/1005/1006 as rate limits
        and back off accordingly.
        """
        import asyncio as _asyncio
        payload = {"model": self.model, "texts": batch, "type": self.embed_type}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    resp = await client.post(
                        f"{self.base_url}/embeddings",
                        json=payload,
                        headers=headers,
                    )
                    if resp.status_code == 429:
                        # Standard rate-limit response
                        retry_after = float(resp.headers.get("Retry-After", 0)) or (
                            self.retry_base_delay * (2 ** attempt)
                        )
                        await _asyncio.sleep(retry_after)
                        last_exc = EmbeddingError(f"rate limited (attempt {attempt + 1})")
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                # MiniMax shape: {"vectors": [[...]], "base_resp": {...}}
                base = data.get("base_resp", {}) or {}
                base_code = base.get("status_code", 0)
                # Rate-limit codes per MiniMax docs
                if base_code in (1002, 1003, 1004, 1005, 1006):
                    # Rate limit / quota / RPM / RPS. Exponential backoff.
                    delay = self.retry_base_delay * (2 ** attempt)
                    await _asyncio.sleep(delay)
                    last_exc = EmbeddingError(
                        f"minimax base_resp status={base_code} msg={base.get('status_msg', '')[:120]}"
                    )
                    continue
                if base_code and base_code != 0:
                    # Non-rate-limit API error — fail fast
                    raise EmbeddingError(
                        f"MiniMax error: status_code={base_code} "
                        f"msg={base.get('status_msg', '')[:200]}"
                    )
                vectors = data.get("vectors")
                if not vectors:
                    raise EmbeddingError(f"MiniMax returned no vectors: {data}")
                return vectors
            except httpx.HTTPError as exc:
                last_exc = exc
                await _asyncio.sleep(self.retry_base_delay * (2 ** attempt))
        raise EmbeddingError(f"MiniMax embed failed after {self.max_retries} attempts: {last_exc}")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            vectors = await self._embed_batch(batch)
            if vectors and self.dim != len(vectors[0]):
                self.dim = len(vectors[0])
            results.extend(vectors)
        return results

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]


async def auto_detect_provider(prefer: str | None = None) -> EmbeddingProvider:
    """Pick the best available provider based on env vars + reachability.

    Priority (per Duckets 2026-06-23): LM Studio primary, MiniMax fallback.
    Sentence-transformers is the offline last resort.

    Order:
      1. DUCKBOT_EMBEDDING env var (explicit) — honored over prefer
      2. `prefer` argument (e.g. "lmstudio") — used if DUCKBOT_EMBEDDING unset
      3. LM Studio reachable → LMStudioEmbeddings (DEFAULT)
      4. MINIMAX_API_KEY set → MiniMax (FALLBACK)
      5. OPENAI_API_KEY set → OpenAI (alt fallback)
      6. sentence-transformers installed → LocalEmbeddings
      7. None available → raises EmbeddingError
    """
    explicit = os.environ.get("DUCKBOT_EMBEDDING", "").lower().strip()
    if explicit == "openai":
        return OpenAIEmbeddings()
    if explicit == "minimax":
        return MiniMaxEmbeddings()
    if explicit == "lmstudio":
        return LMStudioEmbeddings()
    if explicit == "local":
        return LocalEmbeddings()

    target = (prefer or "").lower().strip() or "lmstudio"

    def _make_lmstudio():
        lm_url = os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234/v1")
        return LMStudioEmbeddings(base_url=lm_url)

    def _make_minimax():
        return MiniMaxEmbeddings()

    def _make_openai():
        return OpenAIEmbeddings()

    # Try the preferred target first, then walk fallback chain
    chain = []
    if target == "lmstudio":
        chain = [_make_lmstudio, _make_minimax, _make_openai]
    elif target == "minimax":
        chain = [_make_minimax, _make_lmstudio, _make_openai]
    elif target == "openai":
        chain = [_make_openai, _make_lmstudio, _make_minimax]
    else:
        chain = [_make_lmstudio, _make_minimax, _make_openai]

    last_exc = None
    for factory in chain:
        try:
            return factory()
        except EmbeddingError as e:
            last_exc = e
            continue

    # Try local sentence-transformers
    try:
        import sentence_transformers  # noqa: F401
        return LocalEmbeddings()
    except ImportError:
        pass

    raise EmbeddingError(
        f"No embedding provider available. Last error: {last_exc}"
    )


def make_query_embedder(ingest_embedder: EmbeddingProvider) -> EmbeddingProvider:
    """Return a fresh provider configured for query-time embedding.

    For MiniMax, "type=query" uses a different retrieval-optimized pass
    that produces better results when matching queries against the
    "type=db" pass used during ingest.

    For other providers, returns a copy with no behavior change.
    """
    if isinstance(ingest_embedder, MiniMaxEmbeddings):
        import copy
        q = copy.copy(ingest_embedder)
        q.embed_type = "query"
        return q
    return ingest_embedder


def get_default_provider() -> EmbeddingProvider:
    """Sync variant of auto_detect_provider. Use only in sync contexts."""
    import asyncio
    return asyncio.run(auto_detect_provider())


async def is_lmstudio_reachable(url: str | None = None, timeout: float = 2.0) -> bool:
    """Quick check: is LM Studio responding (with or without auth)?"""
    base = (url or os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234/v1")).rstrip("/v1")
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(f"{base}/v1/models")
            # 200 (no auth) or 401 (auth required) both mean "server is up"
            return r.status_code in (200, 401)
    except Exception:
        return False


__all__ = [
    "EmbeddingProvider",
    "EmbeddingError",
    "OpenAIEmbeddings",
    "LMStudioEmbeddings",
    "LocalEmbeddings",
    "MiniMaxEmbeddings",
    "auto_detect_provider",
    "get_default_provider",
    "is_lmstudio_reachable",
    "make_query_embedder",
]
