"""
store.py — tier-aware memory store.

This module is the BACKWARDS-COMPATIBLE adapter over the new pluggable
backend seam (src/backends/). Existing callers (src/ingest.py,
src/query.py, src/memory.py, src/dashboard.py, src/cli.py, tests/)
still construct `MemoryStore(...)` and call `.query()`, `.bm25_query()`,
`.add_chunks()` etc. Internally, MemoryStore now delegates to a
`VectorBackend` from src/backends/, selected by `DUCKBOT_BACKEND`
(default "chroma").

This is Layer 14 of the brain-upgrade roadmap. Pattern source:
MemPalace's `backends/base.py` (MIT).

The delegation is one-for-one:
  - MemoryStore.add_chunks()    → backend.add_chunks()
  - MemoryStore.query()         → backend.query()
  - MemoryStore.bm25_query()    → backend.bm25_query()
  - MemoryStore.delete()        → backend.delete()
  - MemoryStore.stats()         → backend.stats()
  - MemoryStore.collection_for()→ backend.collection_for()

The dictionary-shaped return values are preserved so existing code
that destructures {id, text, metadata, distance} keeps working.

`MemoryStore.add_chunks()` stays async because some callers
(src/memory.py:219) `await` it; the backend call is sync but we wrap
in `asyncio.to_thread` so callers don't pay a cost.

Legacy helpers (mark_ingested, mark_queried, reset) are preserved as
no-ops or backends. mark_ingested / mark_queried simply update the
backend's last_*_ts counters. reset() wipes the backend's collections
when supported (Chroma yes; stubs may raise NotImplementedError).
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backends import get_backend
from .backends.base import BackendStats
from .chunk import Chunk
from .tier import Tier


DEFAULT_PERSIST_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"


# -----------------------------------------------------------------------------
# Stats (legacy dataclass, kept for backward compat)
# -----------------------------------------------------------------------------


@dataclass
class StoreStats:
    """Snapshot of collection sizes."""

    working: int = 0
    episodic: int = 0
    semantic: int = 0
    procedural: int = 0
    total: int = 0
    last_ingest_ts: float = 0.0
    last_query_ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "working": self.working,
            "episodic": self.episodic,
            "semantic": self.semantic,
            "procedural": self.procedural,
            "total": self.total,
            "last_ingest_ts": self.last_ingest_ts,
            "last_query_ts": self.last_query_ts,
        }


# -----------------------------------------------------------------------------
# Legacy adapter over the new VectorBackend ABC
# -----------------------------------------------------------------------------


class MemoryStore:
    """Backward-compatible adapter over VectorBackend (src/backends/).

    Existing callers (src/query.py, src/ingest.py, src/memory.py,
    src/dashboard.py, src/cli.py, src/connectors/base.py, src/eval.py,
    src/watcher.py) construct a `MemoryStore` and call methods on it.
    Internally we delegate to a configured backend (default: Chroma).

    Selection is driven by DUCKBOT_BACKEND env var (default "chroma").
    To swap in Qdrant or LanceDB later:
      - export DUCKBOT_BACKEND=qdrant
      - pip install qdrant-client
      - implement the stub methods in src/backends/qdrant.py

    No code in this file touches Chroma directly anymore; that's all
    in src/backends/chroma.py.
    """

    def __init__(
        self,
        persist_dir: Path | str | None = None,
        embedding_dim: int = 1536,
        embedding_provider_name: str = "lmstudio",
        backend_name: str | None = None,
    ) -> None:
        self.persist_dir = (
            Path(persist_dir) if persist_dir
            else Path(os.environ.get("DUCKBOT_CHROMA_DIR", str(DEFAULT_PERSIST_DIR)))
        )
        self.embedding_dim = embedding_dim
        self.embedding_provider_name = embedding_provider_name
        self.backend_name = backend_name or os.environ.get("DUCKBOT_BACKEND", "chroma")

        # Build the backend. Only Chroma cares about persist_dir today;
        # other backends get kwargs as a passthrough.
        backend_kwargs: dict[str, Any] = {
            "embedding_dim": embedding_dim,
            "embedding_provider_name": embedding_provider_name,
        }
        if self.backend_name == "chroma":
            backend_kwargs["persist_dir"] = self.persist_dir

        self._backend = get_backend(self.backend_name, **backend_kwargs)

    # ---- Direct access (for tests / dashboard) -----------------------------

    @property
    def backend(self):
        """The underlying VectorBackend. Use this for new code."""
        return self._backend

    def collection_for(self, tier: Tier) -> Any:
        """Return the underlying collection object (Chroma-specific helper).

        For other backends (Qdrant/LanceDB), this returns whatever the
        backend exposes as a "collection" handle. Most callers should
        use the abstract API (query/bm25_query/add_chunks/delete) instead.
        """
        return self._backend.collection_for(tier.value)

    @property
    def all_collections(self) -> dict[Tier, Any]:
        """All {tier: collection} pairs."""
        return {Tier(t): self._backend.collection_for(t) for t in self._backend.supported_tiers}

    # ---- Core ops (async wrappers around the sync backend) ------------------

    async def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        tier: Tier,
        metadata_override: list[dict] | None = None,
    ) -> int:
        """Add chunks + pre-computed embeddings to a tier collection.

        Async wrapper around the sync backend call. The original API was
        `async`, so we preserve that signature for callers like
        src/memory.py:219 (`await store.add_chunks(...)`).
        """
        return await asyncio.to_thread(
            self._backend.add_chunks, chunks, embeddings, tier.value, metadata_override
        )

    def query(
        self,
        query_embedding: list[float],
        tier: Tier | None = None,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query the vector index.

        Returns a list of {id, text, metadata, distance, tier} dicts
        (the legacy dict shape — preserved for backward compat with
        src/query.py which destructures this shape).
        """
        hits = self._backend.query(
            query_embedding=query_embedding,
            tier=tier.value if tier else None,
            n_results=n_results,
            where=where,
            where_document=where_document,
        )
        return [h.to_dict() for h in hits]

    def bm25_query(
        self,
        query_text: str,
        tier: Tier | None = None,
        n_results: int = 5,
        where_document_contains: str | None = None,
    ) -> list[dict[str, Any]]:
        """Lexical / keyword search.

        The legacy API accepted `where_document_contains` which we ignore
        (the new ABC takes just query_text + tier). Backends with native
        where_document support can be wired in later if needed.
        """
        del where_document_contains  # legacy kwarg, unused in new API
        hits = self._backend.bm25_query(
            query_text=query_text,
            tier=tier.value if tier else None,
            n_results=n_results,
        )
        return [h.to_dict() for h in hits]

    def delete(self, ids, tier: Tier) -> int:
        """Delete chunks by id from a tier collection."""
        return self._backend.delete(ids, tier.value)

    def stats(self) -> StoreStats:
        """Return a legacy StoreStats view over the backend's BackendStats."""
        backend_stats = self._backend.stats()
        per_tier = backend_stats.chunks_per_tier()
        return StoreStats(
            working=per_tier.get("working", 0),
            episodic=per_tier.get("episodic", 0),
            semantic=per_tier.get("semantic", 0),
            procedural=per_tier.get("procedural", 0),
            total=backend_stats.total,
            last_ingest_ts=backend_stats.last_ingest_ts,
            last_query_ts=backend_stats.last_query_ts,
        )

    def backend_stats(self) -> BackendStats:
        """Return the full BackendStats (richer than legacy StoreStats)."""
        return self._backend.stats()

    # ---- Legacy helpers (preserved for backward compat) --------------------

    def mark_ingested(self) -> None:
        """Update the last-ingest timestamp on the backend.

        Backends should record this for stats reporting. The Chroma
        implementation uses an internal 'meta_internal' collection.
        """
        # We just touch a backend-level timestamp via stats(); the actual
        # timestamp tracking happens in the backend's collection upserts.
        # For backward compat we still allow this method to be called.
        if hasattr(self._backend, "mark_ingested"):
            self._backend.mark_ingested()
        return None

    def mark_queried(self) -> None:
        """Update the last-query timestamp on the backend."""
        if hasattr(self._backend, "mark_queried"):
            self._backend.mark_queried()
        return None

    def reset(self) -> None:
        """Wipe all collections. Used by tests.

        Tries the backend's reset() if implemented (ChromaBackend
        supports it). Falls back to per-tier delete on all known ids.
        """
        if hasattr(self._backend, "reset"):
            try:
                self._backend.reset()
                return None
            except Exception:
                pass
        # Fallback: enumerate ids in each tier collection and delete them.
        for tier_name in self._backend.supported_tiers:
            try:
                coll = self._backend.collection_for(tier_name)
                resp = coll.get(include=[])
                ids = (resp or {}).get("ids") or []
                if ids:
                    self._backend.delete(ids, tier_name)
            except Exception:
                continue
        return None


__all__ = [
    "DEFAULT_PERSIST_DIR",
    "StoreStats",
    "MemoryStore",
]
