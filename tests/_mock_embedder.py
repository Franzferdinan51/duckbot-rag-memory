"""Mock embedder for tests that don't need real embeddings."""

import asyncio
import hashlib
import math
from typing import Any


def _hash_to_vec(text: str, dim: int = 384) -> list[float]:
    """Deterministic hash-to-vector. Same text -> same vector."""
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Expand to dim by repeating + mixing
    raw = (h * ((dim // len(h)) + 1))[:dim]
    # Normalize to unit vector
    norm = math.sqrt(sum(b * b for b in raw)) or 1.0
    return [b / norm for b in raw]


class MockEmbeddings:
    """Deterministic hash-based embedder. No API calls. No model downloads."""

    name = "mock"
    dim = 384

    def __init__(self, dim: int = 384):
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [_hash_to_vec(t, self.dim) for t in texts]

    async def embed_one(self, text: str) -> list[float]:
        return _hash_to_vec(text, self.dim)
