"""
chroma.py — ChromaDB backend implementing the VectorBackend ABC.

Wraps the existing tier-aware `MemoryStore` so existing call sites
(src/query.py, src/memory.py, src/connectors/*) can keep working
without changes. New code should use `get_backend("chroma")` to
get a VectorBackend instance.

This is the default backend (DUCKBOT_BACKEND unset → "chroma").
MIT (chromadb is Apache-2.0; this wrapper is DuckBot brain, MIT).
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from .base import BackendStats, TierStats, VectorBackend, VectorHit


# Coercion helper (kept identical to src/store.py._coerce_chroma).
def _coerce(value: Any) -> str | int | float | bool:
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(v) for v in value)[:200]
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, default=str)[:200]
    return str(value)[:200]


class ChromaBackend(VectorBackend):
    """VectorBackend backed by ChromaDB. One collection per tier."""

    DEFAULT_PERSIST_DIR = Path("data") / "chroma"

    # Supported distance metrics for HNSW.
    #   "cosine" — default; works for any embedding, normalizes internally.
    #   "l2"     — Euclidean; use when embeddings are not pre-normalized.
    #   "ip"     — inner product; faster, equivalent to cosine ONLY for
    #              pre-normalized vectors (e.g. BGE models with
    #              normalize_embeddings=True at ingest).
    SUPPORTED_DISTANCE_METRICS = ("cosine", "l2", "ip")

    # Max chunks per coll.upsert() call. Larger batches risk segfaulting
    # ChromaDB's hnswlib/sqlite3 native bindings on macOS — verified by
    # ingesting a 696-line MEMORY.md (~60 chunks) which crashed at the
    # old single-call upsert; splitting in half worked. 32 matches the
    # LM Studio embedder batch size so memory + vector writes stay in
    # sync. Override with DUCKBOT_CHROMA_UPSERT_BATCH.
    DEFAULT_UPSERT_BATCH = 32

    def __init__(
        self,
        persist_dir: Optional[Path | str] = None,
        embedding_dim: int = 1536,
        embedding_provider_name: str = "lmstudio",
        tier_names: Optional[list[str]] = None,
        distance_metric: str = "cosine",
    ) -> None:
        # Lazy import so the rest of the package works without chromadb.
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as e:
            raise ImportError(
                "ChromaBackend requires chromadb. pip install chromadb"
            ) from e

        if distance_metric not in self.SUPPORTED_DISTANCE_METRICS:
            raise ValueError(
                f"distance_metric must be one of {self.SUPPORTED_DISTANCE_METRICS}, "
                f"got {distance_metric!r}"
            )

        self._persist_dir = (
            Path(persist_dir) if persist_dir else Path(
                os.environ.get("DUCKBOT_CHROMA_DIR", str(self.DEFAULT_PERSIST_DIR))
            )
        )
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self.embedding_provider_name = embedding_provider_name
        self.distance_metric = distance_metric

        self._tier_names: list[str] = list(tier_names or [
            "working", "episodic", "semantic", "procedural",
        ])

        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collections: dict[str, Any] = {}
        for tier in self._tier_names:
            # NOTE: Chroma's `metadata["hnsw:space"]` only takes effect on
            # collection CREATION. If you change distance_metric on an
            # existing store, you must delete the collection and let it
            # be recreated (or use a new persist_dir).
            self._collections[tier] = self._client.get_or_create_collection(
                name=f"duckbot_{tier}",
                metadata={
                    "hnsw:space": distance_metric,
                    "tier": tier,
                    "embedding_dim": embedding_dim,
                    "embedding_provider": embedding_provider_name,
                },
            )

    # ---- Identity ----------------------------------------------------------

    @property
    def name(self) -> str:
        return "chroma"

    @property
    def supported_tiers(self) -> list[str]:
        return list(self._tier_names)

    @property
    def persist_dir(self) -> Path:
        return self._persist_dir

    # ---- Admin ------------------------------------------------------------
    def reset(self) -> None:
        """Wipe all ChromaDB data and reinitialize empty collections.

        Atomically: (1) close the client, (2) delete the persist dir,
        (3) recreate the client, (4) re-create all tier collections.
        After reset() the store is in a clean initialized state — no chunks,
        no orphaned vectors. Used by tests and the CLI reset command.
        """
        import shutil
        # 1. Close the client so ChromaDB releases file locks.
        #    ChromaDB PersistentClient has no explicit close(); setting to None
        #    releases the reference. On Windows, this is required before rmtree.
        self._client = None
        # 2. Delete the entire persist dir (wipes sqlite + all collection dirs).
        if self._persist_dir.exists():
            shutil.rmtree(str(self._persist_dir), ignore_errors=True)
        # 3. Recreate the persist dir so client re-initializes cleanly.
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        # 4. Re-create the client.
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        self._client = chromadb.PersistentClient(
            path=str(self._persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        # 5. Re-create all tier collections (restores _collections dict).
        for tier in self._tier_names:
            self._get_or_create_collection(tier)

    # ---- Core ops ----------------------------------------------------------

    def add_chunks(
        self,
        chunks: list[Any],
        embeddings: list[list[float]],
        tier: str,
        metadata_override: Optional[list[dict[str, Any]]] = None,
    ) -> int:
        if tier not in self._tier_names:
            raise ValueError(f"unknown tier: {tier!r}; supported: {self._tier_names}")
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunk/embedding count mismatch: {len(chunks)} chunks, "
                f"{len(embeddings)} embeddings"
            )
        if metadata_override is not None and len(metadata_override) != len(chunks):
            raise ValueError(
                f"metadata_override count mismatch: {len(chunks)} chunks, "
                f"{len(metadata_override)} overrides"
            )
        coll = self._collections[tier]
        # Read the upsert batch size once per call so operators can tune
        # without restarting. Clamp to [1, len(chunks)].
        try:
            batch_size = int(os.environ.get("DUCKBOT_CHROMA_UPSERT_BATCH", self.DEFAULT_UPSERT_BATCH))
        except (TypeError, ValueError):
            batch_size = self.DEFAULT_UPSERT_BATCH
        batch_size = max(1, min(batch_size, len(chunks)))

        ids_all = [c.id for c in chunks]
        documents_all = [c.text for c in chunks]
        metadatas_all: list[dict[str, Any]] = []
        for i, c in enumerate(chunks):
            m = {
                "source_path": c.source_path,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
                "has_code": c.has_code,
                "char_count": c.char_count,
                "tier": tier,
                "ingested_at": int(time.time()),
            }
            if getattr(c, "section_header", None):
                m["section_header"] = c.section_header[:200]
            # L13 verbatim-first
            verbatim = getattr(c, "verbatim_text", None) or c.text
            if len(verbatim) > 8192:
                verbatim = verbatim[:8192] + "\n...[truncated]"
            m["verbatim_text"] = verbatim
            if metadata_override is not None:
                m.update(metadata_override[i])
            metadatas_all.append({k: _coerce(v) for k, v in m.items()})

        # v0.15.1: batch the upsert. A single coll.upsert() with hundreds
        # of vectors segfaults ChromaDB's hnswlib/sqlite3 bindings on
        # macOS (verified on a 696-line MEMORY.md). 32 per call matches
        # the LM Studio embedder batch size and keeps each native call
        # under the allocation threshold that triggers the segfault.
        added = 0
        for start in range(0, len(chunks), batch_size):
            end = min(start + batch_size, len(chunks))
            try:
                coll.upsert(
                    ids=ids_all[start:end],
                    embeddings=embeddings[start:end],
                    documents=documents_all[start:end],
                    metadatas=metadatas_all[start:end],
                )
                added += end - start
            except Exception as exc:
                # Surface which batch failed so a caller can retry just
                # that slice instead of the whole ingest. The next batch
                # still attempts — partial-success > total-fail.
                raise RuntimeError(
                    f"chroma upsert failed for tier={tier} batch={start}..{end} "
                    f"(of {len(chunks)} chunks, batch_size={batch_size}): {exc}"
                ) from exc

        # Track the last-ingest timestamp so stats() doesn't have to scan.
        self._last_ingest_ts = time.time()
        return added

    def query(
        self,
        query_embedding: list[float],
        tier: Optional[str] = None,
        n_results: int = 5,
        where: Optional[dict[str, Any]] = None,
        where_document: Optional[dict[str, Any]] = None,
    ) -> list[VectorHit]:
        if tier is not None and tier not in self._tier_names:
            raise ValueError(f"unknown tier: {tier!r}")
        tiers = [tier] if tier else list(self._tier_names)
        # Use ceil so the union of per-tier hits always has at least
        # n_results candidates after merging. Floor division here caused
        # silent under-fetching (e.g. 5 requested, 4 returned across 4 tiers).
        # Honor n_results=0 literally (return nothing) instead of clamping
        # to 1 — that was an off-by-one for callers passing 0.
        if n_results <= 0:
            per_tier = 0
        elif len(tiers) > 1:
            per_tier = math.ceil(n_results / len(tiers))
        else:
            per_tier = n_results
        out: list[VectorHit] = []
        for t in tiers:
            coll = self._collections[t]
            try:
                resp = coll.query(
                    query_embeddings=[query_embedding],
                    n_results=per_tier,
                    where=where,
                    where_document=where_document,
                    include=["documents", "metadatas", "distances"],
                )
            except Exception:
                continue
            if not resp or not resp.get("ids"):
                continue
            ids = resp["ids"][0]
            docs = resp["documents"][0]
            metas = resp["metadatas"][0]
            dists = resp["distances"][0]
            for i, doc_id in enumerate(ids):
                out.append(VectorHit(
                    id=doc_id,
                    text=docs[i],
                    metadata=dict(metas[i] or {}),
                    tier=t,
                    distance=float(dists[i]),
                ))
        out.sort(key=lambda h: h.distance)
        return out[:n_results]

    def bm25_query(
        self,
        query_text: str,
        tier: Optional[str] = None,
        n_results: int = 5,
    ) -> list[VectorHit]:
        keywords = [k for k in query_text.split() if len(k) > 2][:8]
        if not keywords:
            return []
        conditions = [{"$contains": k} for k in keywords[:4]]
        where_doc: dict[str, Any] = conditions[0] if len(conditions) == 1 else {"$or": conditions}
        tiers = [tier] if tier else list(self._tier_names)
        out: list[VectorHit] = []
        for t in tiers:
            coll = self._collections[t]
            try:
                resp = coll.get(
                    where_document=where_doc,
                    include=["documents", "metadatas"],
                    limit=n_results * 2,
                )
            except Exception:
                continue
            if not resp or not resp.get("ids"):
                continue
            for i, doc_id in enumerate(resp["ids"]):
                doc_text = resp["documents"][i].lower()
                hits = sum(1 for k in keywords if k.lower() in doc_text)
                if hits == 0:
                    continue
                out.append(VectorHit(
                    id=doc_id,
                    text=resp["documents"][i],
                    metadata=dict(resp["metadatas"][i] or {}),
                    tier=t,
                    distance=1.0 - (hits / max(len(keywords), 1)),
                ))
        out.sort(key=lambda h: h.distance)
        return out[:n_results]

    def delete(self, ids: Iterable[str], tier: str) -> int:
        if tier not in self._tier_names:
            raise ValueError(f"unknown tier: {tier!r}")
        ids_list = list(ids)
        if not ids_list:
            return 0
        # Check which ids actually exist BEFORE deleting so we can report
        # the real count. The previous implementation always returned
        # len(ids_list), lying about deletions for unknown ids.
        try:
            before = self._collections[tier].get(ids=ids_list, include=[])
            existing = set((before or {}).get("ids") or [])
        except Exception:
            existing = set(ids_list)  # be optimistic if the precheck fails
        self._collections[tier].delete(ids=ids_list)
        return len([i for i in ids_list if i in existing])

    def stats(self) -> BackendStats:
        tier_stats: list[TierStats] = []
        for t in self._tier_names:
            try:
                count = self._collections[t].count()
            except Exception:
                count = 0
            tier_stats.append(TierStats(name=t, chunk_count=int(count)))
        # Prefer the in-memory tracker updated by add_chunks (O(1)).
        # Fall back to a capped metadata scan (max 1000 rows per tier)
        # only if the tracker has never been set — e.g. for a fresh
        # process importing a pre-existing store.
        last_ingest_ts = float(getattr(self, "_last_ingest_ts", 0.0) or 0.0)
        if last_ingest_ts <= 0:
            for t in self._tier_names:
                try:
                    resp = self._collections[t].get(include=["metadatas"], limit=1000)
                    for m in (resp.get("metadatas") or []):
                        ts = float(m.get("ingested_at") or 0)
                        if ts > last_ingest_ts:
                            last_ingest_ts = ts
                except Exception:
                    pass
        return BackendStats(
            backend_name=self.name,
            tiers=tier_stats,
            last_ingest_ts=last_ingest_ts,
            last_query_ts=float(getattr(self, "_last_query_ts", 0.0) or 0.0),
            extra={"persist_dir": str(self._persist_dir)},
        )

    def mark_ingested(self) -> None:
        """Record the last-ingest timestamp in-memory. stats() reads from
        this to avoid an O(N) metadata scan on every dashboard refresh."""
        self._last_ingest_ts = time.time()

    def mark_queried(self) -> None:
        """Record the last-query timestamp in-memory."""
        self._last_query_ts = time.time()

    # ---- Convenience -------------------------------------------------------

    def collection_for(self, tier: str) -> Any:
        """Direct access to the underlying Chroma collection.

        Not part of the ABC; preserved for backward compatibility with
        existing code (e.g. eval scripts) that touches collections directly.
        """
        if tier not in self._tier_names:
            raise ValueError(f"unknown tier: {tier!r}")
        return self._collections[tier]

    def all_collections(self) -> dict[str, Any]:
        """Return the full {tier: collection} map."""
        return dict(self._collections)

    def close(self) -> None:
        # Chroma doesn't have an explicit close; client handles its own cleanup.
        return None


__all__ = ["ChromaBackend"]
