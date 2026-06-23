"""
store.py — ChromaDB wrapper with tier-aware collections.

Architecture: one collection per memory tier (working/episodic/semantic/procedural).
This lets us:
  - Run tier-specific queries (e.g., "show me procedural rules")
  - Apply different eviction policies per tier
  - Maintain separate metadata schemas per tier
  - Track usage stats independently

Borrowed from Cognee's "collections per semantic layer" pattern.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from .chunk import Chunk
from .tier import Tier


DEFAULT_PERSIST_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"


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


class MemoryStore:
    """Tier-aware ChromaDB wrapper.

    Each tier is its own collection. Metadata is stored alongside embeddings
    so we can do hybrid (vector + filter) queries.
    """

    def __init__(
        self,
        persist_dir: Path | str | None = None,
        embedding_dim: int = 1536,
        embedding_provider_name: str = "openai",
    ) -> None:
        self.persist_dir = Path(persist_dir) if persist_dir else DEFAULT_PERSIST_DIR
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_dim = embedding_dim
        self.embedding_provider_name = embedding_provider_name
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        self._collections: dict[Tier, Any] = {}
        for tier in Tier:
            self._collections[tier] = self._client.get_or_create_collection(
                name=f"duckbot_{tier.value}",
                metadata={
                    "hnsw:space": "cosine",
                    "tier": tier.value,
                    "embedding_dim": embedding_dim,
                    "embedding_provider": embedding_provider_name,
                },
            )

    def collection_for(self, tier: Tier) -> Any:
        return self._collections[tier]

    @property
    def all_collections(self) -> dict[Tier, Any]:
        return self._collections

    async def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        tier: Tier,
    ) -> int:
        """Add chunks + pre-computed embeddings to a tier collection.

        Returns the number of chunks added (skipped if IDs already exist).
        """
        if not chunks:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunk/embedding count mismatch: {len(chunks)} chunks, {len(embeddings)} embeddings"
            )
        coll = self._collections[tier]
        ids = [c.id for c in chunks]
        # Build metadata. Keep it flat — Chroma metadata must be primitives.
        metadatas = []
        for c in chunks:
            m = {
                "source_path": c.source_path,
                "chunk_index": c.chunk_index,
                "total_chunks": c.total_chunks,
                "has_code": c.has_code,
                "char_count": c.char_count,
                "tier": tier.value,
                "ingested_at": int(time.time()),
            }
            if c.section_header:
                m["section_header"] = c.section_header[:200]
            metadatas.append(m)
        documents = [c.text for c in chunks]

        # Use upsert so re-ingesting the same chunk is idempotent.
        coll.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return len(chunks)

    def query(
        self,
        query_embedding: list[float],
        tier: Tier | None = None,
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        where_document: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Query one collection (or all if tier=None).

        Returns a list of {id, text, metadata, distance} dicts sorted by relevance.
        """
        results: list[dict[str, Any]] = []
        tiers = [tier] if tier else list(Tier)
        per_tier = max(1, n_results // len(tiers)) if len(tiers) > 1 else n_results

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
                # Collection might be empty; skip silently
                continue
            if not resp or not resp.get("ids"):
                continue
            ids = resp["ids"][0]
            docs = resp["documents"][0]
            metas = resp["metadatas"][0]
            dists = resp["distances"][0]
            for i, doc_id in enumerate(ids):
                results.append({
                    "id": doc_id,
                    "text": docs[i],
                    "metadata": metas[i],
                    "distance": dists[i],
                    "tier": t.value,
                })
        # Sort by distance (cosine; lower = more similar)
        results.sort(key=lambda r: r["distance"])
        return results[:n_results]

    def bm25_query(
        self,
        query_text: str,
        tier: Tier | None = None,
        n_results: int = 5,
        where_document_contains: str | None = None,
    ) -> list[dict[str, Any]]:
        """Keyword/BM25-style search using ChromaDB's where_document filter.

        ChromaDB doesn't have native BM25 (it uses HNSW on dense vectors), but
        we approximate it with simple `contains` filters. For a small corpus
        (50k-100k chunks) this is plenty.
        """
        keywords = [k for k in query_text.split() if len(k) > 2][:8]
        if not keywords:
            return []
        # Build where_document with $or of contains conditions
        conditions = [{"$contains": k} for k in keywords[:4]]
        where_doc: dict[str, Any] = conditions[0] if len(conditions) == 1 else {"$or": conditions}

        results: list[dict[str, Any]] = []
        tiers = [tier] if tier else list(Tier)
        for t in tiers:
            coll = self._collections[t]
            try:
                resp = coll.get(
                    where_document=where_doc,
                    include=["documents", "metadatas"],
                    limit=n_results * 2,  # over-fetch, then rank
                )
            except Exception:
                continue
            if not resp or not resp.get("ids"):
                continue
            # Rank by number of keyword hits
            for i, doc_id in enumerate(resp["ids"]):
                doc_text = resp["documents"][i].lower()
                hits = sum(1 for k in keywords if k.lower() in doc_text)
                if hits == 0:
                    continue
                results.append({
                    "id": doc_id,
                    "text": resp["documents"][i],
                    "metadata": resp["metadatas"][i],
                    "distance": 1.0 - (hits / max(len(keywords), 1)),  # pseudo-distance
                    "tier": t.value,
                    "bm25_hits": hits,
                })
        results.sort(key=lambda r: r.get("bm25_hits", 0), reverse=True)
        return results[:n_results]

    def stats(self) -> StoreStats:
        s = StoreStats()
        for tier in Tier:
            coll = self._collections[tier]
            count = coll.count()
            setattr(s, tier.value, count)
            s.total += count
        # Track last operation timestamps via a meta collection
        try:
            meta = self._client.get_or_create_collection("meta_internal")
            data = meta.get()
            if data and data.get("metadatas"):
                for m in data["metadatas"]:
                    if "last_ingest_ts" in m:
                        s.last_ingest_ts = m["last_ingest_ts"]
                    if "last_query_ts" in m:
                        s.last_query_ts = m["last_query_ts"]
        except Exception:
            pass
        return s

    def mark_ingested(self) -> None:
        meta = self._client.get_or_create_collection("meta_internal")
        meta.upsert(
            ids=["ingest_marker"],
            documents=["ingest_marker"],  # Chroma requires documents OR images
            metadatas=[{"last_ingest_ts": int(time.time()), "total_at_ingest": self.stats().total}],
        )

    def mark_queried(self) -> None:
        meta = self._client.get_or_create_collection("meta_internal")
        meta.upsert(
            ids=["query_marker"],
            documents=["query_marker"],  # Chroma requires documents OR images
            metadatas=[{"last_query_ts": int(time.time())}],
        )

    def reset(self) -> None:
        """Wipe all collections. Used by tests."""
        for tier in Tier:
            try:
                self._client.delete_collection(f"duckbot_{tier.value}")
            except Exception:
                pass
        try:
            self._client.delete_collection("meta_internal")
        except Exception:
            pass
        # Re-create
        for tier in Tier:
            self._collections[tier] = self._client.get_or_create_collection(
                name=f"duckbot_{tier.value}",
                metadata={"hnsw:space": "cosine", "tier": tier.value},
            )