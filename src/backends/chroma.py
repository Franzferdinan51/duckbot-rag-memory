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
        ids = [c.id for c in chunks]
        metadatas = []
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
            m = {k: _coerce(v) for k, v in m.items()}
            metadatas.append(m)
        documents = [c.text for c in chunks]
        coll.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
        return len(chunks)

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
        per_tier = max(1, n_results // len(tiers)) if len(tiers) > 1 else n_results
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
        self._collections[tier].delete(ids=ids_list)
        return len(ids_list)

    def stats(self) -> BackendStats:
        tier_stats: list[TierStats] = []
        last_ingest_ts = 0.0
        for t in self._tier_names:
            try:
                count = self._collections[t].count()
            except Exception:
                count = 0
            tier_stats.append(TierStats(name=t, chunk_count=int(count)))
        # Best-effort last_ingest_ts: scan metadata (slow for big collections).
        # We cap at 1000 chunks per tier for this scan.
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
            last_ingest_ts=float(last_ingest_ts),
            last_query_ts=float(getattr(self, "_last_query_ts", 0.0) or 0.0),
            extra={"persist_dir": str(self._persist_dir)},
        )

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
