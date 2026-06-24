"""
memory.py — unified memory API for DuckBot.

This is the high-level facade that all integrations (file watcher, MCP tool,
OpenClaw skills, CLI) should use. It wraps the lower-level chunk/tier/store
modules and adds:

  - Single `remember()` entry point: ingest + extract + classify + embed + store
  - Single `recall()` entry point: hybrid retrieval with optional tier filter
  - `reflect()` sleep-time consolidation: episodic → semantic distillation
  - `forget()` explicit deletion with provenance
  - `stats()` snapshot for dashboards

Design references:
  - mem0: hook-based auto-capture (we expose remember() for that)
  - Letta: 3-tier (core/recall/archival) — we have 4 tiers (CoALA)
  - Cognee: ECL pipeline (Extract → Cognify → Load) — `remember()` is ECL+store
  - Hermes Agent: FTS5 + LLM hybrid + periodic nudge (we use LM Studio primary,
    MiniMax fallback per Duckets 2026-06-23 directive)

The API is async. All public methods are coroutines. The CLI wraps them in
asyncio.run() for sync entry points.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .chunk import Chunk, chunk_markdown
from .consolidate import extract_facts_from_chunk, deduplicate_facts
from .embeddings import (
    EmbeddingProvider,
    auto_detect_provider,
    is_lmstudio_reachable,
    make_query_embedder,
)
from .ingest import IngestStats
from .query import QueryResult, QueryStats, hybrid_query
from .store import MemoryStore
from .tier import Tier, classify, reclassify_for_working


# -----------------------------------------------------------------------------
# Result types
# -----------------------------------------------------------------------------

@dataclass
class RememberResult:
    """What `remember()` did."""
    text: str
    chunk_id: str
    tier: Tier
    confidence: float
    entities: list[dict] = field(default_factory=list)  # extracted entities
    relationships: list[dict] = field(default_factory=list)  # extracted triples
    importance: float = 0.5  # 0..1
    duration_ms: float = 0.0
    source_path: str = ""
    metadata: dict = field(default_factory=dict)
    provider: str = ""
    stored: bool = True


@dataclass
class MemoryStats:
    total: int = 0
    by_tier: dict[str, int] = field(default_factory=dict)
    by_provider: dict[str, int] = field(default_factory=dict)
    last_remember_ts: float = 0.0
    last_recall_ts: float = 0.0
    last_reflect_ts: float = 0.0
    lmstudio_reachable: bool = False


# -----------------------------------------------------------------------------
# The Memory class — main facade
# -----------------------------------------------------------------------------

class Memory:
    """High-level memory facade.

    Usage:
        mem = Memory()                          # auto-detect LM Studio → MiniMax
        await mem.remember("Today we did X")    # stores with auto tier
        results = await mem.recall("X?", k=5)  # hybrid retrieval
        await mem.reflect()                     # consolidate episodic
        await mem.forget(chunk_id)              # explicit delete
        snap = await mem.stats()                # dashboard snapshot
    """

    def __init__(
        self,
        store: MemoryStore | None = None,
        embedder: EmbeddingProvider | None = None,
        persist_dir: str | None = None,
    ):
        # Lazy-init: build embedder and store on first use, but if user
        # passed them in, use those.
        self._store = store
        self._embedder = embedder
        self._persist_dir = persist_dir

    async def _ensure_initialized(self) -> tuple[MemoryStore, EmbeddingProvider]:
        if self._store is None:
            # Auto-detect embedder first so we can match its dim to the store
            if self._embedder is None:
                self._embedder = await auto_detect_provider()
            # Some providers (LM Studio, local) have lazy dim resolution;
            # do a single test embed so the dim is set before store init.
            try:
                probe = await self._embedder.embed_one("dim probe")
                if probe and self._embedder.dim != len(probe):
                    self._embedder.dim = len(probe)
            except Exception:
                pass  # dim will be whatever the provider's default is
            self._store = MemoryStore(
                persist_dir=self._persist_dir,
                embedding_dim=self._embedder.dim,
                embedding_provider_name=self._embedder.name,
            )
        return self._store, self._embedder

    # -------------------------------------------------------------------------
    # remember() — single entry point for "save this"
    # -------------------------------------------------------------------------
    async def remember(
        self,
        text: str,
        source_path: str = "<remember>",
        metadata: dict | None = None,
        force_tier: Tier | None = None,
    ) -> RememberResult:
        """Store a single memory. Auto-chunks long text, classifies tier,
        extracts entities + relationships, embeds, and stores.

        Args:
          text: the memory content. Markdown encouraged (headers, lists, code).
          source_path: where this came from. Used for tier classification and
            provenance. Use the file path for files, "<remember>" for ad-hoc.
          metadata: arbitrary dict; stored alongside the chunk.
          force_tier: override auto-classification. Use sparingly.

        Returns:
          RememberResult with chunk_id, tier, importance, etc.
        """
        store, embedder = await self._ensure_initialized()
        meta = dict(metadata or {})
        started = time.time()
        meta.setdefault("created_at", time.time())
        meta.setdefault("importance", 0.5)
        meta.setdefault("last_recalled_at", 0.0)
        meta.setdefault("recall_count", 0)
        meta.setdefault("source", source_path)

        # 1. Chunk (markdown-aware, but for short single-shot input we just
        #    produce a single chunk)
        if len(text) < 2000 and "\n" not in text.strip():
            chunks = [Chunk(
                text=text,
                source_path=source_path,
                start_char=0,
                end_char=len(text),
                chunk_index=0,
                total_chunks=1,
            )]
        else:
            chunks = chunk_markdown(text, source_path=source_path, chunk_size=512)

        if not chunks:
            chunks = [Chunk(
                text=text,
                source_path=source_path,
                start_char=0,
                end_char=len(text),
                chunk_index=0,
                total_chunks=1,
            )]

        # 2. Classify tier (per chunk, but most input is single-chunk)
        results: list[RememberResult] = []
        for chunk in chunks:
            if force_tier is not None:
                tier = force_tier
                confidence = 1.0
            else:
                assignment = classify(chunk.source_path, chunk.text)
                assignment = reclassify_for_working(chunk.source_path, assignment)
                tier = assignment.tier
                confidence = assignment.confidence

            # 3. Extract entities + relationships (simple regex pass for v0.1)
            entities, relationships = _extract_entities_and_relations(chunk.text)

            # 4. Score importance (heuristic; v0.1)
            importance = _score_importance(chunk.text, tier, len(entities), len(relationships))

            # 5. Embed
            try:
                vecs = await embedder.embed([chunk.text])
            except Exception as exc:
                # Fallback: try with query_embedder, or skip
                raise

            # 6. Store
            chunk_meta = dict(meta)
            chunk_meta.update({
                "confidence": confidence,
                "importance": importance,
                "entities": [e["name"] for e in entities],
                "relationships_count": len(relationships),
            })
            chunk_id = chunk.id  # content-hash based, idempotent
            added = await store.add_chunks(
                [chunk], vecs, tier, metadata_override=[chunk_meta]
            )
            stored = added > 0

            result = RememberResult(
                text=chunk.text,
                chunk_id=chunk_id,
                tier=tier,
                confidence=confidence,
                entities=entities,
                relationships=relationships,
                importance=importance,
                duration_ms=(time.time() - started) * 1000,
                source_path=chunk.source_path,
                metadata=chunk_meta,
                provider=embedder.name,
                stored=stored,
            )
            results.append(result)

        store.mark_ingested()
        # Bump importance of the most-relevant prior chunks that this memory
        # is semantically similar to (spreading activation, Letta-inspired).
        if len(results) > 0 and len(text) > 50:
            try:
                await self._bump_related(text, embedder, store)
            except Exception:
                pass

        return results[0]  # primary result; ignore the rest for v0.1

    async def _bump_related(
        self,
        text: str,
        embedder: EmbeddingProvider,
        store: MemoryStore,
        top_k: int = 5,
        bump: float = 0.05,
    ) -> None:
        """Find top-k semantically similar prior memories and bump their
        importance + last_recalled_at. This is a simple spreading-activation
        pattern from Letta's memory-block write protocol."""
        try:
            qe = make_query_embedder(embedder)
            results, _ = await hybrid_query(text, store, qe, n_results=top_k, tier=None)
            for r in results:
                tier_obj = Tier(r.tier) if isinstance(r.tier, str) else r.tier
                tier_coll = store.collection_for(tier_obj)
                cur = tier_coll.get(ids=[r.chunk_id], include=["metadatas"])
                if cur and cur["ids"]:
                    md = dict(cur["metadatas"][0])
                    md["importance"] = float(md.get("importance", 0.5)) + bump
                    md["last_recalled_at"] = time.time()
                    md["recall_count"] = int(md.get("recall_count", 0)) + 1
                    tier_coll.update(ids=[r.chunk_id], metadatas=[md])
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # recall() — search
    # -------------------------------------------------------------------------
    async def recall(
        self,
        query: str,
        k: int = 5,
        tier: Tier | str | None = None,
        min_importance: float | None = None,
        rerank: bool | None = None,
        decay: bool | None = None,
    ) -> tuple[list[QueryResult], QueryStats]:
        """Hybrid retrieval with optional tier filter, importance threshold,
        cross-encoder rerank (Layer 7), and Ebbinghaus decay (Layer 8).

        Updates recall_count + last_recalled_at on returned chunks.

        Args:
            query: the search text
            k: top-k results to return
            tier: restrict to one tier (working/episodic/semantic/procedural)
            min_importance: drop chunks below this importance score
            rerank: True/False forces on/off. None reads DUCKBOT_RERANK env.
            decay: True/False forces on/off. None reads DUCKBOT_DECAY env.
        """
        store, embedder = await self._ensure_initialized()
        qe = make_query_embedder(embedder)
        tier_filter = None
        if isinstance(tier, str):
            tier_filter = Tier(tier)
        elif isinstance(tier, Tier):
            tier_filter = tier
        results, stats = await hybrid_query(
            query, store, qe, n_results=k, tier=tier_filter,
            rerank=rerank, decay=decay,
        )

        # Optional importance filter
        if min_importance is not None:
            results = [r for r in results if r.metadata.get("importance", 0.5) >= min_importance]

        # Bump recall counters
        for r in results:
            try:
                tier_obj = Tier(r.tier) if isinstance(r.tier, str) else r.tier
                coll = store.collection_for(tier_obj)
                cur = coll.get(ids=[r.chunk_id], include=["metadatas"])
                if cur and cur["ids"]:
                    md = dict(cur["metadatas"][0])
                    md["recall_count"] = int(md.get("recall_count", 0)) + 1
                    md["last_recalled_at"] = time.time()
                    md["importance"] = min(1.0, float(md.get("importance", 0.5)) + 0.02)
                    coll.update(ids=[r.chunk_id], metadatas=[md])
            except Exception:
                pass

        store.mark_queried()
        return results, stats

    # -------------------------------------------------------------------------
    # reflect() — sleep-time consolidation
    # -------------------------------------------------------------------------
    async def reflect(self, lookback_days: int = 7, max_chunks: int = 200) -> dict:
        """Pull recent episodic chunks, extract facts, dedupe, and (if LLM
        available) promote to semantic tier. This is the 'dream' pass.

        For v0.1 we use regex-based extraction (cheap, runs locally). A future
        version will call LM Studio's qwen3.5-9b for higher-quality extraction.
        """
        store, embedder = await self._ensure_initialized()
        episodic = store.collection_for(Tier.EPISODIC)
        recent = episodic.get(limit=max_chunks, include=["documents", "metadatas"])
        if not recent or not recent.get("ids"):
            return {"scanned": 0, "extracted": 0, "promoted": 0}

        all_facts = []
        for i, cid in enumerate(recent["ids"]):
            facts = extract_facts_from_chunk(
                recent["documents"][i], cid,
                recent["metadatas"][i].get("source_path", "<unknown>"),
            )
            all_facts.extend(facts)

        deduped = deduplicate_facts(all_facts)

        promoted = 0
        if deduped:
            # Promote deduped facts to semantic tier (using the existing
            # remember() pipeline so they get embedded + indexed)
            for f in deduped[:50]:  # cap per reflection
                try:
                    await self.remember(
                        f.text,
                        source_path=f.source_path,
                        metadata={"fact_kind": f.kind, "confidence": f.confidence},
                        force_tier=Tier.SEMANTIC,
                    )
                    promoted += 1
                except Exception:
                    pass

        # Mark reflection time
        try:
            meta = store._client.get_or_create_collection("meta_internal")
            meta.upsert(
                ids=["reflection_marker"],
                documents=["reflection_marker"],
                metadatas=[{"last_reflect_ts": time.time(), "promoted": promoted}],
            )
        except Exception:
            pass

        return {
            "scanned": len(recent["ids"]),
            "extracted": len(all_facts),
            "after_dedup": len(deduped),
            "promoted": promoted,
        }

    # -------------------------------------------------------------------------
    # forget() — explicit delete
    # -------------------------------------------------------------------------
    async def forget(self, chunk_id: str, tier: Tier | str | None = None) -> bool:
        """Delete a specific memory by id. If tier is given, only search that
        tier. Returns True if deleted."""
        store, _ = await self._ensure_initialized()
        if tier is not None:
            tiers = [Tier(tier)] if isinstance(tier, str) else [tier]
        else:
            tiers = list(Tier)
        for t in tiers:
            coll = store.collection_for(t)
            try:
                coll.delete(ids=[chunk_id])
                return True
            except Exception:
                continue
        return False

    # -------------------------------------------------------------------------
    # stats() — dashboard snapshot
    # -------------------------------------------------------------------------
    async def stats(self) -> MemoryStats:
        store, embedder = await self._ensure_initialized()
        ss = store.stats()
        out = MemoryStats(
            total=ss.total,
            by_tier={
                Tier.WORKING.value: ss.working,
                Tier.EPISODIC.value: ss.episodic,
                Tier.SEMANTIC.value: ss.semantic,
                Tier.PROCEDURAL.value: ss.procedural,
            },
            by_provider={embedder.name: ss.total} if embedder else {},
            last_remember_ts=ss.last_ingest_ts,
            last_recall_ts=ss.last_query_ts,
        )
        out.lmstudio_reachable = await is_lmstudio_reachable()
        return out

    # -------------------------------------------------------------------------
    # reset() — wipe
    # -------------------------------------------------------------------------
    async def reset(self) -> None:
        store, _ = await self._ensure_initialized()
        store.reset()


# -----------------------------------------------------------------------------
# Helpers (private)
# -----------------------------------------------------------------------------

def _stable_id(source_path: str, text: str) -> str:
    """Stable chunk id from source + text. Used for idempotent upserts."""
    h = hashlib.sha256(f"{source_path}\x00{text}".encode("utf-8")).hexdigest()[:16]
    return f"mem_{h}"


def _extract_entities_and_relations(text: str) -> tuple[list[dict], list[dict]]:
    """Lightweight entity + relationship extraction. Regex-based, fast, no LLM.

    Returns:
      entities: list of {"name": str, "type": str, "mentions": int}
      relationships: list of {"subject": str, "relation": str, "object": str}
    """
    import re
    entities: dict[str, dict] = {}
    relationships: list[dict] = []

    # People: "Duckets said/did/went/...", "Ryan said/...", "Mr/Ms/Dr <Name>"
    for m in re.finditer(r"\b([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\s+(said|did|went|told|asked|noted|replied|added|typed|installed|set up|prefers|likes|lives)\b", text):
        name = m.group(1)
        entities.setdefault(name, {"name": name, "type": "person", "mentions": 0})
        entities[name]["mentions"] += 1

    # Organizations: "GitHub", "LM Studio", "OpenClaw", etc. — just capitalized nouns
    for m in re.finditer(r"\b([A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){0,2})\b", text):
        name = m.group(1)
        if name in entities:
            entities[name]["mentions"] += 1
        elif len(name) >= 3 and name not in {"The", "This", "That", "When", "What", "Why", "How", "Where"}:
            entities[name] = {"name": name, "type": "org_or_product", "mentions": 1}

    # Locations: simple "lives at X", "address is X"
    for m in re.finditer(r"(?:lives at|address is|based in|location[:\s]+)\s+([A-Z][^.]{3,60})", text):
        loc = m.group(1).strip()
        if loc:
            entities.setdefault(loc, {"name": loc, "type": "location", "mentions": 1})

    # Relationships: "X installed Y", "X set up Y", "X uses Y", "X prefers Y"
    rel_patterns = [
        (r"([A-Z][A-Za-z0-9 ]+?)\s+(installed|set up|configured|built|wrote)\s+([A-Za-z0-9 .\-\_]+?)(?:\.|$|,)", "did_action_to"),
        (r"([A-Z][A-Za-z0-9 ]+?)\s+(uses|chose|prefers|likes|rejected)\s+([A-Za-z0-9 .\-\_]+?)(?:\.|$|,)", "preference"),
        (r"([A-Z][A-Za-z0-9 ]+?)\s+(is|was)\s+(?:a|an|the)\s+([A-Za-z0-9 \-]+)", "identity"),
    ]
    for pat, rel in rel_patterns:
        for m in re.finditer(pat, text):
            sub, _, obj = m.groups()
            sub, obj = sub.strip(), obj.strip()
            if len(sub) > 3 and len(obj) > 2 and len(sub) < 60 and len(obj) < 60:
                relationships.append({"subject": sub, "relation": rel, "object": obj})

    return list(entities.values()), relationships


def _score_importance(text: str, tier: Tier, entity_count: int, rel_count: int) -> float:
    """Heuristic importance score 0..1. Used for ranking and decay."""
    score = 0.3  # base
    # Tier bonuses
    if tier == Tier.PROCEDURAL:
        score += 0.3  # rules persist longest
    elif tier == Tier.SEMANTIC:
        score += 0.2
    elif tier == Tier.WORKING:
        score += 0.1  # short-lived but high salience
    # Length bonus (longer = more context, up to a cap)
    score += min(0.2, len(text) / 5000.0)
    # Entity + relationship richness
    score += min(0.2, (entity_count + rel_count) * 0.05)
    # Caps
    return max(0.0, min(1.0, score))


# -----------------------------------------------------------------------------
# Convenience: sync wrappers for CLI / scripts
# -----------------------------------------------------------------------------

def sync_remember(text: str, **kwargs) -> RememberResult:
    return asyncio.run(Memory().remember(text, **kwargs))


def sync_recall(query: str, **kwargs) -> tuple[list[QueryResult], QueryStats]:
    return asyncio.run(Memory().recall(query, **kwargs))


def sync_reflect(**kwargs) -> dict:
    return asyncio.run(Memory().reflect(**kwargs))


__all__ = [
    "Memory",
    "RememberResult",
    "MemoryStats",
    "sync_remember",
    "sync_recall",
    "sync_reflect",
]
