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
The fallback chain is: DUCKBOT_EMBEDDING env > auto-detect LM Studio > MiniMax > OpenAI.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Protocol

import httpx


# ---------------------------------------------------------------------------
# Shared HTTP client + rate limiter + result cache.
#
# These were added in the v0.11.2 hotfix to fix the LM Studio embedding
# spam reported 2026-06-24. Root causes:
#
#   1. No embed-result cache. Every `brain_decay_status`, `brain_fsrs_review`,
#      and watcher poll re-embedded the same chunks.
#   2. Each call opened a new `httpx.AsyncClient`. With v0.10/v0.11's
#      three concurrent embed paths (Layer 6 OpenClaw, Layer 16 Hermes,
#      MCP server), bursts collided at LM Studio and triggered
#      `ERR_HTTP_HEADERS_SENT`.
#   3. No rate limiter. When a burst arrived, all callers slammed
#      LM Studio's single-threaded HTTP server.
#
# All three are now handled by `_get_http_client()` (singleton),
# `_rate_limiter` (per-process token bucket), and the LRU `_embed_cache`.
# ---------------------------------------------------------------------------

_http_client: httpx.AsyncClient | None = None
# Lazy lock: created inside _get_http_client so it attaches to the event loop
# that first calls it (important for pytest where each test gets its own loop).
_http_client_lock: asyncio.Lock | None = None


async def _get_http_client(timeout: float = 120.0) -> httpx.AsyncClient:
    """Process-wide shared httpx.AsyncClient."""
    global _http_client, _http_client_lock
    if _http_client is not None and not _http_client.is_closed:
        return _http_client
    # Lazily create lock attached to the current event loop.
    if _http_client_lock is None:
        _http_client_lock = asyncio.Lock()
    async with _http_client_lock:
        # Double-check after acquiring the lock.
        if _http_client is not None and not _http_client.is_closed:
            return _http_client
        _http_client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=30.0,
            ),
        )
        return _http_client


async def close_http_client() -> None:
    """Close the shared client. Call from MCP server shutdown.

    Note: does NOT await aclose() on the httpx client — doing so from a
    closed event loop (e.g. pytest's loop-per-test teardown) raises
    RuntimeError. The client is orphaned and GC'd; connections close naturally.
    """
    global _http_client, _http_client_lock
    _http_client = None
    # Drop the stale lock so the next _get_http_client call creates a fresh one
    # attached to whatever event loop is current (critical for pytest reuse).
    _http_client_lock = None


@dataclass
class _TokenBucket:
    """Simple async token-bucket rate limiter."""
    rate_per_min: int = 60
    capacity: int = 60
    _tokens: float = field(init=False, default=0.0)
    _last_refill: float = field(init=False, default=0.0)
    _lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        env_rpm = os.environ.get("DUCKBOT_EMBED_RPM", "").strip()
        if env_rpm:
            try:
                self.rate_per_min = max(1, int(env_rpm))
                self.capacity = self.rate_per_min
            except ValueError:
                pass
        self._tokens = float(self.capacity)
        self._last_refill = time.monotonic()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            refill = (elapsed / 60.0) * self.rate_per_min
            self._tokens = min(float(self.capacity), self._tokens + refill)
            self._last_refill = now
            if self._tokens < 1.0:
                deficit = 1.0 - self._tokens
                wait = (deficit / self.rate_per_min) * 60.0
                await asyncio.sleep(wait)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


_rate_limiter = _TokenBucket()


def reset_rate_limiter(rpm=None) -> None:
    """Reset the global rate limiter. Tests use this; production rarely does."""
    global _rate_limiter
    if rpm is not None:
        _rate_limiter = _TokenBucket(rate_per_min=rpm, capacity=rpm)
    else:
        _rate_limiter = _TokenBucket()


@dataclass
class _EmbedCache:
    """LRU cache for embed results."""
    max_size: int = 4096
    _data: dict = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        env_sz = os.environ.get("DUCKBOT_EMBED_CACHE_SIZE", "").strip()
        if env_sz:
            try:
                self.max_size = max(0, int(env_sz))
            except ValueError:
                pass

    def _key(self, text: str, model: str):
        return (hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest(), model)

    def get(self, text: str, model: str):
        if self.max_size == 0:
            return None
        k = self._key(text, model)
        v = self._data.get(k)
        if v is not None:
            self._data.pop(k, None)
            self._data[k] = v
        return v

    def put(self, text: str, model: str, vec) -> None:
        if self.max_size == 0:
            return
        k = self._key(text, model)
        if k in self._data:
            self._data.pop(k, None)
        elif len(self._data) >= self.max_size:
            try:
                self._data.pop(next(iter(self._data)))
            except StopIteration:
                pass
        self._data[k] = vec

    def clear(self) -> None:
        self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


_embed_cache = _EmbedCache()


def reset_embed_cache() -> None:
    """Clear the global embed cache. Tests use this."""
    global _embed_cache
    _embed_cache = _EmbedCache()


def get_embed_cache_stats() -> dict:
    """Return cache stats for diagnostics. Used by `cli doctor` and tests."""
    return {"size": len(_embed_cache), "max_size": _embed_cache.max_size}


# ---------------------------------------------------------------------------
# Per-endpoint dim cache for LMStudioEmbeddings._resolve_dim().
# Keyed by (base_url, model) so different LM Studio instances + model combos
# each get their own dim resolved independently. A failed probe is cached as
# None so we don't keep retrying a broken endpoint in the same process.
# Added in v0.11.2 alongside _embed_cache to fix "dim probe" spam in LM Studio.
# ---------------------------------------------------------------------------
_EMBEDDER_DIM_CACHE: dict[tuple[str, str], int | None] = {}


def reset_dim_cache() -> None:
    """Clear the per-endpoint dim cache. Tests use this.

    Clears in-place so that any module-level import of _EMBEDDER_DIM_CACHE
    (e.g. `from src.embeddings import _EMBEDDER_DIM_CACHE`) stays valid —
    replacing the reference with `= {}` would leave stale imported refs dirty.
    """
    _EMBEDDER_DIM_CACHE.clear()


def get_dim_cache_stats() -> dict:
    """Return dim cache stats for diagnostics."""
    return {
        "size": len(_EMBEDDER_DIM_CACHE),
        "entries": {str(k): v for k, v in _EMBEDDER_DIM_CACHE.items()},
    }


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
        # Check cache first; only send uncached to the server.
        results: list = [None] * len(texts)  # type: ignore[list-item]
        to_fetch: list = []
        for i, t in enumerate(texts):
            cached = _embed_cache.get(t, self.model)
            if cached is not None:
                results[i] = cached
            else:
                to_fetch.append((i, t))
        if not to_fetch:
            return results  # type: ignore[return-value]
        for i in range(0, len(to_fetch), self.batch_size):
            batch = to_fetch[i:i + self.batch_size]
            batch_texts = [t for _, t in batch]
            payload = {"model": self.model, "input": batch_texts}
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            await _rate_limiter.acquire()
            client = await _get_http_client(timeout=60.0)
            resp = await client.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            vectors = [item["embedding"] for item in data["data"]]
            for (orig_idx, text), vec in zip(batch, vectors):
                results[orig_idx] = vec
                _embed_cache.put(text, self.model, vec)
        return results  # type: ignore[return-value]

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
        """Try to learn the actual embedding dim from a test query.

        v0.11.2: short-circuit when the dim is already known for this
        (base_url, model) in this process. Long-lived daemons
        (Hermes, MCP server, watcher) instantiate LMStudioEmbeddings
        many times — without the cache every dim resolution fires a
        real /v1/embeddings call against LM Studio, which shows up as
        "test" spam in the server log.
        """
        cache_key = (self.base_url, self.model)
        cached = _EMBEDDER_DIM_CACHE.get(cache_key)
        if cached is not None and cached > 0:
            if self.dim != cached:
                self.dim = cached
            return
        try:
            client = await _get_http_client(timeout=5.0)
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
                _EMBEDDER_DIM_CACHE[cache_key] = self.dim
            else:
                # Non-200: server up but wrong model/auth — don't retry this combo
                _EMBEDDER_DIM_CACHE[cache_key] = None
        except Exception:
            # Network/connection error — don't retry
            _EMBEDDER_DIM_CACHE[cache_key] = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list = [None] * len(texts)  # type: ignore[list-item]
        to_fetch: list = []
        for i, t in enumerate(texts):
            cached = _embed_cache.get(t, self.model)
            if cached is not None:
                results[i] = cached
            else:
                to_fetch.append((i, t))
        if not to_fetch:
            return results  # type: ignore[return-value]
        for i in range(0, len(to_fetch), self.batch_size):
            batch = to_fetch[i:i + self.batch_size]
            batch_texts = [t for _, t in batch]
            payload = {"model": self.model, "input": batch_texts}
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            await _rate_limiter.acquire()
            client = await _get_http_client(timeout=120.0)
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
            for (orig_idx, text), vec in zip(batch, vectors):
                results[orig_idx] = vec
                _embed_cache.put(text, self.model, vec)
        return results  # type: ignore[return-value]

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
        """One HTTP call. Raises EmbeddingError on rate-limit or HTTP error."""
        import asyncio as _asyncio
        payload = {"model": self.model, "texts": batch, "type": self.embed_type}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                await _rate_limiter.acquire()
                client = await _get_http_client(timeout=60.0)
                resp = await client.post(
                    f"{self.base_url}/embeddings",
                    json=payload,
                    headers=headers,
                )
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", 0)) or (
                        self.retry_base_delay * (2 ** attempt)
                    )
                    await _asyncio.sleep(retry_after)
                    last_exc = EmbeddingError(f"rate limited (attempt {attempt + 1})")
                    continue
                resp.raise_for_status()
                data = resp.json()
                base = data.get("base_resp", {}) or {}
                base_code = base.get("status_code", 0)
                if base_code in (1002, 1003, 1004, 1005, 1006):
                    delay = self.retry_base_delay * (2 ** attempt)
                    await _asyncio.sleep(delay)
                    last_exc = EmbeddingError(
                        f"minimax base_resp status={base_code} msg={base.get('status_msg', '')[:120]}"
                    )
                    continue
                if base_code and base_code != 0:
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
        results: list = [None] * len(texts)  # type: ignore[list-item]
        to_fetch: list = []
        for i, t in enumerate(texts):
            cached = _embed_cache.get(t, self.model)
            if cached is not None:
                results[i] = cached
            else:
                to_fetch.append((i, t))
        if not to_fetch:
            return results  # type: ignore[return-value]
        for i in range(0, len(to_fetch), self.batch_size):
            batch = to_fetch[i:i + self.batch_size]
            batch_texts = [t for _, t in batch]
            vectors = await self._embed_batch(batch_texts)
            if vectors and self.dim != len(vectors[0]):
                self.dim = len(vectors[0])
            for (orig_idx, text), vec in zip(batch, vectors):
                results[orig_idx] = vec
                _embed_cache.put(text, self.model, vec)
        return results  # type: ignore[return-value]

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
    "close_http_client",
    "reset_embed_cache",
    "reset_rate_limiter",
    "get_embed_cache_stats",
    "_EMBEDDER_DIM_CACHE",
    "reset_dim_cache",
    "get_dim_cache_stats",
]
