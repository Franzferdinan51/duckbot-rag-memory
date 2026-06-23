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
from dataclasses import dataclass
from typing import Protocol

import httpx


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
            self.model = os.environ.get("LMSTUDIO_MODEL", "text-embedding-nomic-embed-text-v1.5")
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

    Uses the minimax-portal MiniMax embeddings endpoint (compatible with
    OpenAI /v1/embeddings shape).

    Configure via:
      MINIMAX_API_KEY     — required
      MINIMAX_BASE_URL    — default https://api.minimax.io/v1
      MINIMAX_EMBED_MODEL — default text-embedding-01 (1536d)
    """

    model: str = "text-embedding-01"
    name: str = "minimax"
    dim: int = 1536
    api_key: str = ""
    base_url: str = ""
    batch_size: int = 100

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
            if vectors and self.dim != len(vectors[0]):
                self.dim = len(vectors[0])
            results.extend(vectors)
        return results

    async def embed_one(self, text: str) -> list[float]:
        return (await self.embed([text]))[0]


async def auto_detect_provider() -> EmbeddingProvider:
    """Pick the best available provider based on env vars + reachability.

    Order:
      1. DUCKBOT_EMBEDDING env var (explicit)
      2. OPENAI_API_KEY set → OpenAI
      3. MINIMAX_API_KEY set → MiniMax
      4. LM Studio reachable → LMStudioEmbeddings
      5. sentence-transformers installed → LocalEmbeddings
      6. None available → raises EmbeddingError
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

    # Auto-detect chain
    if os.environ.get("OPENAI_API_KEY"):
        return OpenAIEmbeddings()
    if os.environ.get("MINIMAX_API_KEY"):
        return MiniMaxEmbeddings()

    # Try LM Studio
    lm_url = os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234/v1")
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{lm_url.rstrip('/v1')}/v1/models")
            # 200 = no-auth server, 401 = auth required (still reachable)
            if r.status_code in (200, 401):
                return LMStudioEmbeddings(base_url=lm_url)
    except Exception:
        pass

    # Try local sentence-transformers
    try:
        import sentence_transformers  # noqa: F401
        return LocalEmbeddings()
    except ImportError:
        pass

    raise EmbeddingError(
        "No embedding provider available. Set DUCKBOT_EMBEDDING=openai|minimax|lmstudio|local "
        "or install sentence-transformers for offline mode."
    )


def get_default_provider() -> EmbeddingProvider:
    """Sync variant of auto_detect_provider. Use only in sync contexts."""
    import asyncio
    return asyncio.run(auto_detect_provider())


__all__ = [
    "EmbeddingProvider",
    "EmbeddingError",
    "OpenAIEmbeddings",
    "LMStudioEmbeddings",
    "LocalEmbeddings",
    "MiniMaxEmbeddings",
    "auto_detect_provider",
    "get_default_provider",
]
