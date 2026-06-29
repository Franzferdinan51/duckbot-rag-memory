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
  - Hermes Agent: FTS5 + periodic nudge (DuckBot keeps chat-model work in
    the agent; embeddings/rerank stay local-first here)

The API is async. All public methods are coroutines. The CLI wraps them in
asyncio.run() for sync entry points.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

from .chunk import Chunk, chunk_markdown
from .consolidate import extract_facts_from_chunk, deduplicate_facts
from .embeddings import (
    EmbeddingProvider,
    auto_detect_provider,
    is_lmstudio_reachable,
    make_query_embedder,
    _EMBEDDER_DIM_CACHE,
)
from .ingest import IngestStats
from .query import QueryResult, QueryStats, hybrid_query
from .store import MemoryStore
from .tier import Tier, classify, reclassify_for_working, coerce_optional_tier


# -----------------------------------------------------------------------------
# Process-level cache for the embedding-dim probe.
#
# When Memory() is constructed it has to learn the real dim of the active
# embedding model — the defaults in src/embeddings.py are guesses (e.g.
# 1024 for lmstudio). The probe used to be `embed_one("dim probe")` on
# every Memory() instantiation, which fires a real /v1/embeddings call
# against LM Studio. In a long-lived process (Hermes gateway, MCP server,
# watcher, OpenClaw extension) Memory() is re-instantiated on every tool
# call, so the probe spammed LM Studio's request log with "dim probe"
# entries and triggered ERR_HTTP_HEADERS_SENT noise.
#
# The fix: cache the resolved dim by (base_url, model) at module scope,
# so the probe runs at most once per process per model. Failed probes
# are cached as None so we don't keep retrying on a broken endpoint.
# -----------------------------------------------------------------------------
_DIM_PROBE_CACHE: dict[tuple[str, str], int | None] = {}


# Process-wide singleton Memory. Memory() is constructed lazily (the
# embedder + Chroma PersistentClient are only built on first
# _ensure_initialized call), so sharing a single instance is safe and
# fixes a real memory leak: every call used to open a new PersistentClient
# (SQLite + native file handles + SentenceTransformer workers), and the
# watcher instantiates Memory() once per poll cycle, so an idle watcher
# would accumulate file handles + worker threads indefinitely.
#
# The cached instance is ONLY used when no args are passed (the common
# case: `mem = Memory()` in CLI/MCP handlers). When a caller passes
# store=/embedder=/persist_dir=, a fresh instance is created so we
# don't accidentally route through someone else's embedder.
_DEFAULT_MEMORY: "Memory" | None = None  # type: ignore[name-defined]


def reset_default_memory() -> None:
    """Drop the cached singleton Memory. Used by tests that want to
    inject a fake Memory class — without this, the cached instance from
    a previous test would be returned and bypass the monkeypatch.
    Also safe to call at process shutdown as a defense-in-depth cleanup.
    """
    global _DEFAULT_MEMORY
    _DEFAULT_MEMORY = None


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

    def __new__(cls, *args, **kwargs):
        # Singleton fast path: the common case is `Memory()` (no args).
        # Caching the no-arg instance fixes the watcher leak where
        # every poll cycle opened a new Chroma PersistentClient.
        if not args and not kwargs:
            global _DEFAULT_MEMORY
            if _DEFAULT_MEMORY is None:
                _DEFAULT_MEMORY = super().__new__(cls)
            return _DEFAULT_MEMORY
        return super().__new__(cls)

    def __init__(
        self,
        store: MemoryStore | None = None,
        embedder: EmbeddingProvider | None = None,
        persist_dir: str | None = None,
    ):
        # If a no-arg call returned the cached singleton, skip re-init
        # so the lazy embedder + store survive across calls. When the
        # caller passed args (custom store / embedder / persist_dir),
        # Python's __new__ always returned a fresh instance — re-init it.
        global _DEFAULT_MEMORY
        # Use getattr: when __new__ returns the cached singleton,
        # __init__ is invoked but the instance attributes haven't been set
        # yet — accessing self._store would raise AttributeError.
        if (self is _DEFAULT_MEMORY
                and (getattr(self, "_store", None) is not None
                     or getattr(self, "_embedder", None) is not None)):
            # Cached singleton has already been initialized in a previous
            # call. Keep the existing state.
            return
        # Lazy-init: build embedder and store on first use, but if user
        # passed them in, use those.
        self._store = store
        self._embedder = embedder
        self._persist_dir = persist_dir
        # _write_lock is created lazily (inside remember()) on first use
        # rather than here in __init__ to avoid requiring an asyncio event
        # loop at construction time. Tests that instantiate Memory()
        # synchronously (no running loop) would otherwise fail on Python 3.9
        # with "There is no current event loop in thread 'MainThread'."

    async def _ensure_initialized(self) -> tuple[MemoryStore, EmbeddingProvider]:
        if self._store is None:
            # Auto-detect embedder first so we can match its dim to the store
            if self._embedder is None:
                self._embedder = await auto_detect_provider()
            # v0.11.2: skip the probe entirely if we've already learned the
            # dim for this (base_url, model) in this process.
            # _EMBEDDER_DIM_CACHE is the source of truth (populated here and
            # by LMStudioEmbeddings._resolve_dim). _DIM_PROBE_CACHE syncs
            # from it for backward compat with existing code that reads it.
            probe_key = (
                getattr(self._embedder, "base_url", ""),
                getattr(self._embedder, "model", ""),
            )
            # Three states:
            #   key absent       → never probed, do it now
            #   key → int (>0)  → cached dim, apply it
            #   key → None      → previously failed, do NOT retry
            if probe_key in _EMBEDDER_DIM_CACHE:
                cached_dim = _EMBEDDER_DIM_CACHE[probe_key]
                if cached_dim is not None and cached_dim > 0:
                    if self._embedder.dim != cached_dim:
                        self._embedder.dim = cached_dim
                # else: cached None (failed probe) → keep default dim, don't retry
            else:
                try:
                    probe = await self._embedder.embed_one("dim probe")
                    if probe:
                        actual = len(probe)
                        if self._embedder.dim != actual:
                            self._embedder.dim = actual
                        # Populate both caches so LMStudioEmbeddings and tests
                        # both see the resolved dim without re-probing.
                        _EMBEDDER_DIM_CACHE[probe_key] = actual
                        _DIM_PROBE_CACHE[probe_key] = actual
                except Exception:
                    # Mark as failed so we don't keep retrying
                    _EMBEDDER_DIM_CACHE[probe_key] = None
                    _DIM_PROBE_CACHE[probe_key] = None
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
        facts: list[str] | None = None,
    ) -> RememberResult:
        """Store a single memory. Auto-chunks long text, classifies tier,
        extracts entities + relationships, embeds, and stores.

        Args:
          text: the memory content. Markdown encouraged (headers, lists, code).
          source_path: where this came from. Used for tier classification and
            provenance. Use the file path for files, "<remember>" for ad-hoc.
          metadata: arbitrary dict; stored alongside the chunk.
          force_tier: override auto-classification. Use sparingly.
          facts: optional pre-extracted durable facts the agent already pulled
            out of `text`. When provided, each fact is stored as its own
            semantic-tier chunk with metadata.kind="agent_fact" — keeping
            fact extraction in the agent's hands (OpenClaw / Hermes / your
            LLM) rather than forcing the brain to load a separate model.
            The brain itself uses regex heuristics for fully-autonomous mode.

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
        # 0. Spellcheck (MemPalace-inspired, opt-in via DUCKBOT_SPELLCHECK).
        #    Default ON for short single-shot input; OFF for long markdown
        #    since markdown has lots of code blocks / proper nouns that
        #    we don't want to mangle.
        spellcheck_enabled = os.environ.get("DUCKBOT_SPELLCHECK", "1") not in ("0", "false", "no")
        if spellcheck_enabled and len(text) < 2000 and "\n" not in text.strip():
            from .spellcheck import fix_text
            text = fix_text(text)

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
                # Coerce string → Tier for caller convenience (the MCP and CLI
                # surfaces both pass tier as a string from JSON-RPC). Without
                # this, `force_tier="episodic"` would crash at store.add_chunks
                # with `'str' object has no attribute 'value'`.
                tier = coerce_optional_tier(force_tier)
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

            # 5.5 Conflict detection (mem0-inspired, Apache 2.0).
            #
            # If a near-duplicate (cosine > 0.92) already exists in the same
            # tier, we DON'T blindly upsert. Instead, mark the existing chunk
            # as `superseded_by` pointing at the new chunk_id, and stamp the
            # new chunk with `supersedes` pointing back. This preserves the
            # audit trail — old recall results still resolve, but new queries
            # prefer the fresh fact.
            #
            # Cheap: one vector query against the same tier. Cost = one
            # embedding (already computed above) + one Chroma call.
            #
            # SERIALIZED via self._write_lock: the conflict-detection +
            # add_chunks sequence is a check-then-update that must not race.
            # Without this lock, two parallel ingests of the same fact both
            # see "no existing chunk" and both write without marking
            # superseded_by — leaving stale duplicates in the graph.
            # 6. Store
            # Build chunk_meta BEFORE the lock so the fallback path can
            # access it on lock failure. The dict is mutated inside the
            # lock (supersedes key) but the base shape is set here.
            chunk_meta = dict(meta)
            chunk_meta.update({
                "confidence": confidence,
                "importance": importance,
                "entities": [e["name"] for e in entities],
                "relationships_count": len(relationships),
            })
            chunk_id = chunk.id  # content-hash based, idempotent
            # `added` is set inside the lock block (success path) or in the
            # except branch (lock failure fallback). `stored` is True if the
            # chunk was added (added > 0 means new chunk_id didn't exist
            # before). Initialize to 0 so the except branch can assign.
            added = 0
            # Lazily create the write lock on first use (avoids requiring an
            # asyncio event loop at Memory.__init__ time on Python 3.9).
            if not hasattr(self, "_write_lock"):
                import asyncio
                self._write_lock = asyncio.Lock()
            try:
                async with self._write_lock:
                    tier_coll_pre = store.collection_for(tier)
                    existing = tier_coll_pre.query(
                        query_embeddings=[vecs[0]],
                        n_results=1,
                        include=["metadatas", "distances"],
                    )
                    if existing and existing.get("ids") and existing["ids"][0]:
                        eid = existing["ids"][0][0]
                        edist = existing["distances"][0][0]
                        emeta = (existing.get("metadatas") or [[{}]])[0][0] or {}
                        # Chroma returns cosine DISTANCE; 0 = identical, 2 = opposite.
                        # 0.08 distance ≈ 0.92 similarity — the standard near-dup bar.
                        already_superseded = bool(emeta.get("superseded_by"))
                        if edist < 0.08 and not already_superseded and eid != chunk.id:
                            # Mark old as superseded; new chunk stores the backref.
                            emeta["superseded_by"] = chunk.id
                            emeta["superseded_at"] = time.time()
                            tier_coll_pre.update(ids=[eid], metadatas=[emeta])
                            chunk_meta["supersedes"] = eid
                    added = await store.add_chunks(
                        [chunk], vecs, tier, metadata_override=[chunk_meta]
                    )
            except Exception as exc:
                # If the lock or store fails for any reason, fall back to
                # an unlocked add_chunks so the ingest isn't dropped entirely.
                # The lock is for conflict detection; the actual write is
                # idempotent (content-hash chunk_id) so a racing duplicate
                # is recoverable.
                logger.warning("ingest under write_lock failed (%s); falling back", exc)
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

        # Agent-provided facts: each gets its own semantic-tier chunk so
        # reflect() can promote them without re-extracting. We do NOT load a
        # chat model ourselves — extraction stays in the agent's hands.
        #
        # Batched: a single embed() call for all facts (vs. the previous
        # N separate calls) + a single add_chunks() call (vs. N). For 10
        # facts this is ~20x fewer HTTP round-trips. _bump_related is
        # skipped for agent_facts (they're already linked via
        # source_chunk_id, and bumping would be N more embed calls).
        if facts:
            try:
                await self._store_agent_facts(
                    facts, source_path, results[0].chunk_id, embedder, store,
                )
            except Exception as exc:
                logger.warning("agent fact batch store failed: %s", exc)

        return results[0]  # primary result; ignore the rest for v0.1

    async def _store_agent_facts(
        self,
        facts: list[str],
        source_path: str,
        parent_chunk_id: str,
        embedder: EmbeddingProvider,
        store: "MemoryStore",
    ) -> int:
        """Batched store for agent-extracted facts.

        Single embed() + single add_chunks() instead of N+1 of each. With
        10 facts, that's 20+ HTTP round-trips collapsed to 2.
        Returns the number of facts successfully stored.
        """
        # Clean, dedupe, and gate (5-char min, 300-char max).
        seen: set[str] = set()
        clean: list[str] = []
        for f in facts:
            t = (f or "").strip()
            if not t or len(t) < 5 or len(t) > 300:
                continue
            if t in seen:
                continue
            seen.add(t)
            clean.append(t)
        if not clean:
            return 0

        # ONE embed call for the whole batch.
        vecs = await embedder.embed(clean)
        if len(vecs) != len(clean):
            logger.warning(
                "agent facts: embed returned %d vectors for %d texts",
                len(vecs), len(clean),
            )

        # Build chunk + metadata once, then ONE add_chunks() call.
        from .chunk import Chunk
        now = time.time()
        chunks = []
        metas = []
        for i, text in enumerate(clean):
            cid = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            chunk_id = f"{cid}-{i}"
            chunks.append(Chunk(
                text=text,
                source_path=source_path,
                start_char=0,
                end_char=len(text),
                chunk_index=i,
                total_chunks=len(clean),
            ))
            metas.append({
                "kind": "agent_fact",
                "source_chunk_id": parent_chunk_id,
                "agent_extracted": True,
                "importance": 0.5,
                "created_at": now,
                "ingested_at": now,
                "last_recalled_at": 0.0,
                "recall_count": 0,
            })
        await store.add_chunks(
            chunks,
            [vecs[i] if i < len(vecs) else vecs[0] for i in range(len(chunks))],
            Tier.SEMANTIC,
            metadata_override=metas,
        )
        return len(chunks)

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
        tier_priors: bool | None = None,
        tier_priors_overrides: dict[str, float] | None = None,
        fsrs: bool | None = None,
    ) -> tuple[list[QueryResult], QueryStats]:
        """Hybrid retrieval with optional tier filter, importance threshold,
        cross-encoder rerank (Layer 7), Ebbinghaus decay (Layer 8), tier
        priors (Layer 11), and FSRS-6 spaced repetition (Layer 9).

        Updates recall_count + last_recalled_at on returned chunks.

        Empty/whitespace queries raise ValueError rather than returning 5
        random semantically-similar chunks. (The MCP server already
        validates this; the Python API matches it now.)

        Args:
            query: the search text
            k: top-k results to return
            tier: restrict to one tier (working/episodic/semantic/procedural)
            min_importance: drop chunks below this importance score
            rerank: True/False forces on/off. None reads DUCKBOT_RERANK env.
            decay: True/False forces on/off. None reads DUCKBOT_DECAY env.
            tier_priors: True/False forces on/off. None reads
                DUCKBOT_TIER_PRIORS env. Per-tier multiplicative weights:
                procedural=1.5, semantic=1.2, episodic=1.0, working=0.8.
            tier_priors_overrides: optional dict mapping tier name -> prior.
            fsrs: True/False forces on/off. None reads DUCKBOT_FSRS env.
                Replaces Ebbinghaus retention with the FSRS-6 power-law
                forgetting curve; per-chunk stability_days + difficulty.
        """
        store, embedder = await self._ensure_initialized()
        # Reject empty/whitespace queries — they'd return random semantically-
        # similar chunks and waste a round-trip to the embedder. Match the
        # MCP handler's validation so the Python API behaves consistently.
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        qe = make_query_embedder(embedder)
        tier_filter = None
        tier_filter = coerce_optional_tier(tier)
        results, stats = await hybrid_query(
            query, store, qe, n_results=k, tier=tier_filter,
            rerank=rerank, decay=decay,
            tier_priors=tier_priors,
            tier_priors_overrides=tier_priors_overrides,
            fsrs=fsrs,
        )

        # Optional importance filter
        if min_importance is not None:
            results = [r for r in results if r.metadata.get("importance", 0.5) >= min_importance]

        # Bump recall counters
        for r in results:
            try:
                tier_obj = coerce_optional_tier(r.tier)
                if tier_obj is None:
                    continue
                coll = store.collection_for(tier_obj)
                cur = coll.get(ids=[r.chunk_id], include=["metadatas"])
                if cur and cur["ids"]:
                    md = dict(cur["metadatas"][0])
                    md["recall_count"] = int(md.get("recall_count", 0)) + 1
                    md["last_recalled_at"] = time.time()
                    md["importance"] = min(1.0, float(md.get("importance", 0.5)) + 0.02)
                    # Bump FSRS stability on every successful recall — this is
                    # the core "memories strengthen when you use them" loop.
                    # Without it, recall_count goes up but forgetting math
                    # stays the same, so the brain never learns from usage.
                    from .decay import bump_stability
                    cur_s = md.get("fsrs_stability_days") or md.get("stability_days")
                    md["fsrs_stability_days"] = bump_stability(
                        float(cur_s) if cur_s is not None else None,
                        recalled=True,
                    )
                    md["fsrs_last_review_ts"] = time.time()
                    coll.update(ids=[r.chunk_id], metadatas=[md])
            except Exception:
                pass

        store.mark_queried()
        return results, stats

    # -------------------------------------------------------------------------
    # reflect() — sleep-time consolidation
    # -------------------------------------------------------------------------
    async def reflect(
        self,
        lookback_days: int = 7,
        max_chunks: int = 200,
        *,
        extract_callback=None,
    ) -> dict:
        """Pull recent episodic chunks, extract facts, dedupe, and promote
        them to semantic tier. This is the 'dream' pass.

        Fact extraction stays in the agent's hands:
          - If `extract_callback` is provided, the brain calls it for each
            chunk: callback(chunk_text, chunk_id, source_path) -> list[str]
            The strings are stored as semantic-tier chunks.
          - Otherwise the brain uses lightweight regex heuristics
            (no model load).

        Pass `extract_callback=lambda text, cid, sp: [...]` from the
        agent (OpenClaw / Hermes) to plug in your own extractor without
        forcing the brain to load a chat model.
        """
        store, embedder = await self._ensure_initialized()
        episodic = store.collection_for(Tier.EPISODIC)
        cutoff = None
        get_kwargs = {
            "limit": max_chunks,
            "include": ["documents", "metadatas"],
        }
        if lookback_days and lookback_days > 0:
            cutoff = time.time() - (lookback_days * 86400)
            get_kwargs["where"] = {"ingested_at": {"$gte": cutoff}}
        try:
            recent = episodic.get(**get_kwargs)
        except TypeError:
            # Some collection adapters may not support metadata filters.
            recent = episodic.get(limit=max_chunks, include=["documents", "metadatas"])
        if not recent or not recent.get("ids"):
            return {"scanned": 0, "extracted": 0, "promoted": 0}

        ids = list(recent.get("ids") or [])
        docs = list(recent.get("documents") or [])
        metadatas = list(recent.get("metadatas") or [])

        if cutoff is not None:
            filtered: list[tuple[str, str, dict]] = []
            for cid, doc, md in zip(ids, docs, metadatas):
                ts = 0.0
                if isinstance(md, dict):
                    ts = float(md.get("ingested_at") or md.get("created_at") or 0.0)
                if ts >= cutoff:
                    filtered.append((cid, doc, md or {}))
            if filtered:
                ids, docs, metadatas = [list(col) for col in zip(*filtered)]
            else:
                ids, docs, metadatas = [], [], []

        if not ids:
            return {"scanned": 0, "extracted": 0, "promoted": 0}

        all_facts = []
        for i, cid in enumerate(ids):
            src_path = (metadatas[i] or {}).get("source_path", "<unknown>")
            agent_facts: list[str] | None = None
            if extract_callback is not None:
                try:
                    agent_facts = extract_callback(docs[i], cid, src_path)
                except Exception as exc:
                    logger.warning("extract_callback failed for %s: %s", cid, exc)
                    agent_facts = None
            facts = extract_facts_from_chunk(
                docs[i], cid, src_path, agent_facts=agent_facts,
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
            "scanned": len(ids),
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
        tier_obj = coerce_optional_tier(tier)
        tiers = [tier_obj] if tier_obj is not None else list(Tier)
        for t in tiers:
            coll = store.collection_for(t)
            try:
                # Chroma's delete is a no-op for unknown ids, so check
                # presence first to report an accurate True/False.
                existing = coll.get(ids=[chunk_id], include=[])
                if not existing or not existing.get("ids"):
                    continue
                coll.delete(ids=[chunk_id])
                return True
            except Exception:
                continue
        return False

    # -------------------------------------------------------------------------
    # supersede() — mark chunk as replaced by a new one (2026-06-29)
    # -------------------------------------------------------------------------
    async def supersede(
        self,
        old_chunk_id: str,
        new_chunk_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> dict:
        """Mark `old_chunk_id` as superseded by `new_chunk_id` (or just by
        reason if no new chunk). Does NOT delete the old chunk — it stays
        in the store with `metadata.superseded_by` + `metadata.superseded_reason`
        + `metadata.superseded_at` so the audit trail is preserved.

        Why not just delete: a fresh agent might re-import the same wrong fact
        from a doc, or two contradicting memories might both be retrieved and
        the agent needs to know which wins. The supersede marker is the
        authoritative answer.

        Returns: {superseded: bool, old_chunk_id, new_chunk_id, reason}.
        """
        import time as _time
        store, _ = await self._ensure_initialized()
        now = _time.time()
        marker = {
            "superseded_by": new_chunk_id or reason or "manual",
            "superseded_at": now,
        }
        if reason:
            marker["superseded_reason"] = reason

        found = False
        for t in Tier:
            coll = store.collection_for(t)
            try:
                existing = coll.get(ids=[old_chunk_id], include=["metadatas"])
                if not existing or not existing.get("ids"):
                    continue
                # Existing metadatas are flattened dicts; merge our marker in.
                old_meta = existing["metadatas"][0] if existing.get("metadatas") else {}
                new_meta = {**(old_meta or {}), **marker}
                coll.update(ids=[old_chunk_id], metadatas=[new_meta])
                found = True
                break
            except Exception:
                continue
        return {
            "superseded": found,
            "old_chunk_id": old_chunk_id,
            "new_chunk_id": new_chunk_id,
            "reason": reason,
        }

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
