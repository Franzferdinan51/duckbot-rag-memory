"""
base.py — abstract base class for vector backends.

This is Layer 14 of the brain-upgrade roadmap (docs/RESEARCH.md).
Pattern source: MemPalace's `backends/base.py` (MIT).

The contract is intentionally narrow so any backend (ChromaDB, Qdrant,
LanceDB, pgvector, Weaviate, ...) can be plugged in by implementing
five methods. Existing code (MemoryStore, query pipeline, MCP server)
talks only to this ABC.

The five methods:
  - add_chunks(chunks, embeddings, tier, metadata_override)
  - query(query_embedding, tier, n_results, where, where_document)
  - bm25_query(query_text, tier, n_results)
  - delete(ids, tier)  # soft-delete by id
  - stats() → BackendStats

Two backends ship now:
  - ChromaBackend (current MemoryStore wrapped; MIT)
  - QdrantBackend / LanceDBBackend — stubs that raise NotImplementedError
    until their native deps are installed (qdrant-client / pylance).

Design rules:
  - All methods are synchronous. The Memory facade runs them in a
    thread pool if it needs to be async.
  - Returned VectorHit objects are JSON-safe (no opaque refs).
  - Stats are per-tier. The Memory facade aggregates them.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


# -----------------------------------------------------------------------------
# Result types (JSON-safe)
# -----------------------------------------------------------------------------


@dataclass
class VectorHit:
    """One retrieval hit. JSON-safe, no opaque refs."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    tier: str = "unknown"
    distance: float = 1.0  # 0 = identical, higher = less similar

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": dict(self.metadata or {}),
            "tier": self.tier,
            "distance": float(self.distance),
        }


@dataclass
class TierStats:
    """Per-tier statistics."""

    name: str
    chunk_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "chunk_count": self.chunk_count}


@dataclass
class BackendStats:
    """Aggregate stats across all tiers."""

    backend_name: str
    tiers: list[TierStats] = field(default_factory=list)
    last_ingest_ts: float = 0.0
    last_query_ts: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return sum(t.chunk_count for t in self.tiers)

    def chunks_per_tier(self) -> dict[str, int]:
        return {t.name: t.chunk_count for t in self.tiers}

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend_name": self.backend_name,
            "tiers": [t.to_dict() for t in self.tiers],
            "total": self.total,
            "chunks_per_tier": self.chunks_per_tier(),
            "last_ingest_ts": self.last_ingest_ts,
            "last_query_ts": self.last_query_ts,
            "extra": dict(self.extra or {}),
        }


# -----------------------------------------------------------------------------
# Backend ABC
# -----------------------------------------------------------------------------


class VectorBackend(abc.ABC):
    """Abstract base for a tier-aware vector backend.

    Every concrete backend must implement the five core methods below.
    Helper properties (name, tiers) carry the metadata that the
    registry uses to dispatch.
    """

    # ---- Identity ----------------------------------------------------------

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short identifier used in DUCKBOT_BACKEND and stats output."""
        ...

    @property
    @abc.abstractmethod
    def supported_tiers(self) -> list[str]:
        """List of tier names this backend is configured for.

        Usually ["working", "episodic", "semantic", "procedural"] but
        a backend can choose to alias them (e.g. one combined collection).
        """
        ...

    # ---- Core operations ---------------------------------------------------

    @abc.abstractmethod
    def add_chunks(
        self,
        chunks: list[Any],  # list[Chunk]
        embeddings: list[list[float]],
        tier: str,
        metadata_override: Optional[list[dict[str, Any]]] = None,
    ) -> int:
        """Upsert chunks with pre-computed embeddings into a tier collection.

        Args:
            chunks: list of Chunk dataclasses. Chunk has .id, .text,
                .source_path, .verbatim_text, .section_header, etc.
            embeddings: parallel list of pre-computed embedding vectors.
            tier: which tier to add to (must be in supported_tiers).
            metadata_override: optional parallel list of metadata dicts
                to merge on top of the auto-generated metadata. Used by
                the Memory facade to inject importance, entities,
                recall_count, etc.

        Returns:
            Number of chunks upserted.
        """
        ...

    @abc.abstractmethod
    def query(
        self,
        query_embedding: list[float],
        tier: Optional[str] = None,
        n_results: int = 5,
        where: Optional[dict[str, Any]] = None,
        where_document: Optional[dict[str, Any]] = None,
    ) -> list[VectorHit]:
        """Query the vector index.

        Args:
            query_embedding: 1-D embedding vector of the query.
            tier: if set, restrict to one tier. If None, query all tiers.
            n_results: maximum hits to return across all tiers.
            where: optional metadata filter (backend-specific dialect).
            where_document: optional document-text filter (backend-specific).

        Returns:
            List of VectorHit sorted by relevance (most-similar first).
        """
        ...

    @abc.abstractmethod
    def bm25_query(
        self,
        query_text: str,
        tier: Optional[str] = None,
        n_results: int = 5,
    ) -> list[VectorHit]:
        """Lexical/BM25-style query.

        Args:
            query_text: the raw query string.
            tier: optional tier filter.
            n_results: max hits.

        Returns:
            List of VectorHit, ranked by keyword-match score.

        Note: ChromaDB doesn't have native BM25; the ChromaBackend
        approximates with `where_document contains` and ranks by hit
        count. Qdrant ships with built-in BM25 (sparse vectors) once
        a sparse model is configured.
        """
        ...

    @abc.abstractmethod
    def delete(self, ids: Iterable[str], tier: str) -> int:
        """Delete chunks by id.

        Args:
            ids: chunk ids to delete.
            tier: tier collection to delete from.

        Returns:
            Number of chunks deleted.

        Note: callers should rarely need this — we keep memory
        append-only. But the seam must support it for tests,
        GDPR-style erasure, and decay eviction.
        """
        ...

    @abc.abstractmethod
    def stats(self) -> BackendStats:
        """Return current backend stats."""
        ...

    # ---- Optional hooks ----------------------------------------------------

    def close(self) -> None:
        """Optional cleanup hook. Default no-op."""
        return None


# -----------------------------------------------------------------------------
# Plugin registration
# -----------------------------------------------------------------------------


# Allow external code to register additional backends at runtime without
# modifying src/backends/__init__.py. Pattern: write a small adapter module
# and call register_backend("my_name", "my.module.MyBackend").
_EXTRA_REGISTRY: dict[str, str] = {}


def register_backend(name: str, fully_qualified_class_path: str) -> None:
    """Register an additional backend at runtime.

    Args:
        name: short name (used in DUCKBOT_BACKEND).
        fully_qualified_class_path: e.g. "my_pkg.adapters.MyBackend".
    """
    if not name or not isinstance(name, str):
        raise ValueError("backend name must be a non-empty string")
    if not fully_qualified_class_path or "." not in fully_qualified_class_path:
        raise ValueError("class path must be fully qualified, e.g. 'pkg.mod.ClassName'")
    _EXTRA_REGISTRY[name] = fully_qualified_class_path


def all_known_backends() -> dict[str, str]:
    """Return built-in + runtime-registered backends."""
    from . import _REGISTRY, _EXTRA_REGISTRY  # type: ignore
    return {**_REGISTRY, **_EXTRA_REGISTRY}


__all__ = [
    "VectorHit",
    "TierStats",
    "BackendStats",
    "VectorBackend",
    "register_backend",
    "all_known_backends",
]
