"""
qdrant.py — Qdrant stub implementing the VectorBackend ABC.

Qdrant supports native sparse + dense vectors (BM25 + cosine in one
collection) and ships with built-in payload filtering, replication,
and HNSW + scalar quantization. License: Apache-2.0.

This is a stub: importing it will tell you to install qdrant-client.
Until then, the registry will raise a clear ImportError if you try
to use it. We do NOT depend on qdrant-client at install time so
the default `chroma` backend stays zero-dependency-overhead.

Pattern source: MemPalace's `backends/qdrant.py` (MIT, stub pattern).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from .base import BackendStats, VectorBackend, VectorHit


# Lazy-import sentinel — raise with a helpful message on first use.
def _require_qdrant():
    try:
        import qdrant_client  # noqa: F401
        from qdrant_client import QdrantClient  # noqa: F401
        from qdrant_client.http import models as qmodels  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "QdrantBackend requires qdrant-client. "
            "Install with: pip install qdrant-client"
        ) from e


class QdrantBackend(VectorBackend):
    """Qdrant-backed VectorBackend. Stub: requires qdrant-client."""

    def __init__(
        self,
        url: Optional[str] = None,
        path: Optional[Path | str] = None,
        api_key: Optional[str] = None,
        embedding_dim: int = 1536,
        tier_names: Optional[list[str]] = None,
        collection_prefix: str = "duckbot_",
    ) -> None:
        _require_qdrant()
        from qdrant_client import QdrantClient

        self._url = url
        self._path = Path(path) if path else None
        self._api_key = api_key
        self.embedding_dim = embedding_dim
        self._tier_names = list(tier_names or [
            "working", "episodic", "semantic", "procedural",
        ])
        self._collection_prefix = collection_prefix
        if path is not None:
            self._path.mkdir(parents=True, exist_ok=True)
            self._client = QdrantClient(path=str(self._path))
        else:
            self._client = QdrantClient(url=url or "http://localhost:6333", api_key=api_key)
        self._ensure_collections()

    def _ensure_collections(self) -> None:
        # Qdrant collections are created on first use. We do not eagerly
        # create them so that read-only access doesn't write to the server.
        return None

    def _collection_name(self, tier: str) -> str:
        return f"{self._collection_prefix}{tier}"

    @property
    def name(self) -> str:
        return "qdrant"

    @property
    def supported_tiers(self) -> list[str]:
        return list(self._tier_names)

    def add_chunks(self, chunks, embeddings, tier, metadata_override=None) -> int:
        raise NotImplementedError(
            "QdrantBackend.add_chunks is a stub. Install qdrant-client and "
            "implement using QdrantClient.upsert() with named vectors."
        )

    def query(
        self,
        query_embedding,
        tier=None,
        n_results=5,
        where=None,
        where_document=None,
    ) -> list[VectorHit]:
        raise NotImplementedError(
            "QdrantBackend.query is a stub. Use QdrantClient.search() "
            "with Filter(must=[FieldCondition(...)]) when implementing."
        )

    def bm25_query(self, query_text, tier=None, n_results=5) -> list[VectorHit]:
        raise NotImplementedError(
            "QdrantBackend.bm25_query is a stub. Qdrant supports BM25 via "
            "sparse vectors — generate with a BM25 encoder, upsert into a "
            "named 'sparse' vector, then search both vectors and RRF-merge."
        )

    def delete(self, ids, tier) -> int:
        raise NotImplementedError(
            "QdrantBackend.delete is a stub. Use QdrantClient.delete() with "
            "PointsSelector(points=[PointIdsList(...)]) when implementing."
        )

    def stats(self) -> BackendStats:
        raise NotImplementedError(
            "QdrantBackend.stats is a stub. Use QdrantClient.get_collections() "
            "and per-collection VectorParams to assemble BackendStats."
        )

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            return None


__all__ = ["QdrantBackend"]
