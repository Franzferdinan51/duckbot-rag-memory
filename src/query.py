"""
query.py — hybrid retrieval with Reciprocal Rank Fusion (RRF).

Pattern from Cognee + LangChain hybrid retriever:
  1. Vector search (semantic similarity)
  2. BM25/keyword search (lexical match)
  3. Reciprocal Rank Fusion: combine ranks, not raw scores
  4. Optional rerank pass (skipped for v0.1)

RRF formula: score(d) = Σ 1/(k + rank_i(d)) for each retriever i.
k=60 is the standard constant (Cormack et al. 2009).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .embeddings import EmbeddingProvider
from .store import MemoryStore
from .tier import Tier


RRF_K = 60  # standard constant from Cormack et al. 2009


@dataclass
class QueryResult:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    tier: str
    rrf_score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None
    vector_distance: float | None = None
    bm25_hits: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "tier": self.tier,
            "rrf_score": self.rrf_score,
            "vector_rank": self.vector_rank,
            "bm25_rank": self.bm25_rank,
            "vector_distance": self.vector_distance,
            "bm25_hits": self.bm25_hits,
        }


@dataclass
class QueryStats:
    query: str
    vector_results: int = 0
    bm25_results: int = 0
    fused_results: int = 0
    duration_seconds: float = 0.0
    tiers_queried: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "vector_results": self.vector_results,
            "bm25_results": self.bm25_results,
            "fused_results": self.fused_results,
            "duration_seconds": round(self.duration_seconds, 3),
            "tiers_queried": self.tiers_queried,
        }


def _rrf_score(rank: int | None, k: int = RRF_K) -> float:
    """Reciprocal Rank Fusion score. rank=1 → 1/(k+1). Higher = better."""
    if rank is None:
        return 0.0
    return 1.0 / (k + rank)


async def hybrid_query(
    query_text: str,
    store: MemoryStore,
    embedder: EmbeddingProvider,
    n_results: int = 5,
    tier: Tier | None = None,
    over_fetch: int = 3,
) -> tuple[list[QueryResult], QueryStats]:
    """Run hybrid vector + BM25 query, fuse with RRF.

    Args:
        query_text: The user's question.
        store: MemoryStore with loaded ChromaDB.
        embedder: EmbeddingProvider (OpenAI or local).
        n_results: Number of final results to return.
        tier: Restrict to a specific tier (or None for all).
        over_fetch: Fetch this many × n_results from each retriever, then
            fuse + truncate. Improves recall at small cost.

    Returns:
        (results, stats) — sorted by RRF score desc.
    """
    started = time.time()
    stats = QueryStats(query=query_text)
    n_fetch = n_results * over_fetch

    # Phase 1: embed the query
    query_embedding = await embedder.embed_one(query_text)

    # Phase 2: vector search
    vector_hits = store.query(
        query_embedding=query_embedding,
        tier=tier,
        n_results=n_fetch,
    )
    stats.vector_results = len(vector_hits)
    stats.tiers_queried = sorted({h["tier"] for h in vector_hits})

    # Phase 3: BM25/keyword search
    bm25_hits = store.bm25_query(query_text, tier=tier, n_results=n_fetch)
    stats.bm25_results = len(bm25_hits)

    # Phase 4: RRF fusion
    by_id: dict[str, QueryResult] = {}

    for rank, hit in enumerate(vector_hits, start=1):
        cid = hit["id"]
        if cid not in by_id:
            by_id[cid] = QueryResult(
                chunk_id=cid,
                text=hit["text"],
                metadata=hit.get("metadata", {}),
                tier=hit.get("tier", "unknown"),
                rrf_score=0.0,
                vector_distance=hit.get("distance"),
            )
        by_id[cid].vector_rank = rank
        by_id[cid].rrf_score += _rrf_score(rank)

    for rank, hit in enumerate(bm25_hits, start=1):
        cid = hit["id"]
        if cid not in by_id:
            by_id[cid] = QueryResult(
                chunk_id=cid,
                text=hit["text"],
                metadata=hit.get("metadata", {}),
                tier=hit.get("tier", "unknown"),
                rrf_score=0.0,
                bm25_hits=hit.get("bm25_hits"),
            )
        by_id[cid].bm25_rank = rank
        by_id[cid].rrf_score += _rrf_score(rank)
        if by_id[cid].bm25_hits is None or hit.get("bm25_hits", 0) > by_id[cid].bm25_hits:
            by_id[cid].bm25_hits = hit.get("bm25_hits")

    # Phase 5: sort by RRF desc, return top n
    results = sorted(by_id.values(), key=lambda r: r.rrf_score, reverse=True)[:n_results]
    stats.fused_results = len(results)
    stats.duration_seconds = time.time() - started
    store.mark_queried()
    return results, stats


def format_results(results: list[QueryResult], max_chars: int = 400) -> str:
    """Format results for display (or for stuffing into an LLM context window)."""
    if not results:
        return "No results."
    parts = []
    for i, r in enumerate(results, 1):
        preview = r.text[:max_chars]
        if len(r.text) > max_chars:
            preview += "..."
        parts.append(
            f"[{i}] (tier={r.tier}, rrf={r.rrf_score:.4f}"
            + (f", vec_rank={r.vector_rank}" if r.vector_rank else "")
            + (f", bm25_rank={r.bm25_rank}" if r.bm25_rank else "")
            + ")\n"
            + f"Source: {r.metadata.get('source_path', '?')}\n"
            + f"Section: {r.metadata.get('section_header', '(none)')}\n"
            + f"{preview}\n"
        )
    return "\n---\n".join(parts)


__all__ = ["hybrid_query", "QueryResult", "QueryStats", "format_results"]