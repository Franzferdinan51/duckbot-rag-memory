"""
lancedb.py — LanceDB stub implementing the VectorBackend ABC.

LanceDB is a columnar vector DB built on Lance (Apache-2.0). Strengths:
embedded mode (no server), native IVF-PQ + HNSW indexes, easy to back
up (just copy the directory). Good fit for single-machine local memory.

This is a stub: importing it will tell you to install lancedb.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from .base import BackendStats, VectorBackend, VectorHit


def _require_lancedb():
    try:
        import lancedb  # noqa: F401
        import pyarrow as pa  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "LanceDBBackend requires lancedb + pyarrow. "
            "Install with: pip install lancedb pyarrow"
        ) from e


class LanceDBBackend(VectorBackend):
    """LanceDB-backed VectorBackend. Stub: requires lancedb + pyarrow."""

    def __init__(
        self,
        uri: Optional[Path | str] = None,
        embedding_dim: int = 1536,
        tier_names: Optional[list[str]] = None,
    ) -> None:
        _require_lancedb()
        import lancedb

        self._uri = str(uri) if uri else str(
            Path(__file__).resolve().parent.parent / "data" / "lancedb"
        )
        Path(self._uri).mkdir(parents=True, exist_ok=True)
        self._client = lancedb.connect(self._uri)
        self.embedding_dim = embedding_dim
        self._tier_names = list(tier_names or [
            "working", "episodic", "semantic", "procedural",
        ])

    @property
    def name(self) -> str:
        return "lancedb"

    @property
    def supported_tiers(self) -> list[str]:
        return list(self._tier_names)

    def add_chunks(self, chunks, embeddings, tier, metadata_override=None) -> int:
        raise NotImplementedError(
            "LanceDBBackend.add_chunks is a stub. Use lancedb.Table.add() "
            "with a PyArrow schema (vector: list<float>, text: string, "
            "metadata: string-JSON) when implementing."
        )

    def query(self, query_embedding, tier=None, n_results=5, where=None, where_document=None) -> list[VectorHit]:
        raise NotImplementedError(
            "LanceDBBackend.query is a stub. Use lancedb.Table.search() "
            ".where() .limit() .to_list() when implementing."
        )

    def bm25_query(self, query_text, tier=None, n_results=5) -> list[VectorHit]:
        raise NotImplementedError(
            "LanceDBBackend.bm25_query is a stub. LanceDB has full-text "
            "search (since v0.3) — use Table.search().where() with an FTS "
            "index when implementing."
        )

    def delete(self, ids, tier) -> int:
        raise NotImplementedError(
            "LanceDBBackend.delete is a stub. Use Table.delete(f\"id IN {ids}\") "
            "when implementing."
        )

    def stats(self) -> BackendStats:
        raise NotImplementedError(
            "LanceDBBackend.stats is a stub. Use lancedb.table_names() and "
            "per-table .count_rows() to assemble BackendStats."
        )

    def close(self) -> None:
        # LanceDB embedded mode doesn't have an explicit close.
        return None


__all__ = ["LanceDBBackend"]
