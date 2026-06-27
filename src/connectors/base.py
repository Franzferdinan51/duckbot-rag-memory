"""
connectors/base.py — shared facade for the brain that all connectors build on.

This is the contract: any framework connector (OpenClaw, Hermes, future ones)
should call this facade rather than poking at the lower-level modules directly.
The facade gives a single, sync, framework-agnostic interface that maps cleanly
to MCP tools, Python function calls, or CLI subcommands.

Why sync? Connectors are wrappers. The underlying Memory class is async, but
each connector has its own way of bridging sync/async (asyncio.run for CLI,
FastMCP's auto-async for MCP, hermes' threading model for Python import).
We expose a sync API here and let each connector handle the event loop.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from ..graph import Graph, DEFAULT_GRAPH_PATH, Entity, Relationship
from ..blocks import BlockStore, DEFAULT_BLOCKS_PATH
from ..injection_scan import InjectionScanner, QuarantineStore, DEFAULT_QUARANTINE_PATH


# -----------------------------------------------------------------------------
# Async/sync bridging
# -----------------------------------------------------------------------------


def _run_async(coro):
    """Run an async coroutine from sync code without conflicting with a parent loop.

    - If there's no running event loop (CLI / tests / sync entry points):
      use `asyncio.run`, which is the standard idiom.
    - If we ARE inside a running event loop (the MCP server's `mcp_stdio`
      handler, FastMCP, Jupyter, asyncio pytest): nest a new loop in a
      worker thread. This avoids `RuntimeError: asyncio.run() cannot be
      called from a running event loop` and keeps the Brain methods
      callable from both sync and async contexts.

    For higher-throughput async callers (MCP), it's better to expose the
    coroutine and `await` it directly. That's a future refactor; for now,
    every connector goes through the sync Brain facade.

    Note on test-suite flakes (v0.13.0): the previous version used
    `asyncio.run(coro)` from a worker thread. Across many tests, the
    OS-level asyncio loop tracking gets corrupted ("Event loop is
    closed" errors in unrelated tests). The fix: own the loop lifecycle
    explicitly with `new_event_loop()` + `loop.close()`, and serialize
    calls with a module-level lock so two concurrent _run_async calls
    don't fight over a single event loop.
    """
    import concurrent.futures
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — safe to use asyncio.run. Still wrap in a
        # try/finally so a partial-cleanup state doesn't bleed across
        # tests in the same process.
        return asyncio.run(coro)
    # We're in a loop. Run the coroutine in a worker thread with its
    # own dedicated event loop. The lock prevents two concurrent
    # _run_async calls from racing on the same event loop / executor.
    with _RUN_ASYNC_LOCK:
        def _runner() -> object:
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_runner).result()


import threading
_RUN_ASYNC_LOCK = threading.Lock()


# -----------------------------------------------------------------------------
# Result types — JSON-serializable for transport
# -----------------------------------------------------------------------------

@dataclass
class BrainStats:
    """Combined snapshot across all 5 layers."""
    total: int = 0
    vector_chunks: int = 0
    vector_by_tier: dict = field(default_factory=dict)
    graph_entities: int = 0
    graph_relationships: int = 0
    graph_active_relationships: int = 0
    blocks: int = 0
    quarantine_total: int = 0
    quarantine_pending: int = 0
    quarantine_approved: int = 0
    quarantine_rejected: int = 0
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RememberResult:
    """Result of a remember() call, transport-friendly."""
    chunk_id: Optional[str] = None
    tier: Optional[str] = None
    confidence: float = 0.0
    importance: float = 0.0
    entities: list = field(default_factory=list)
    relationships: list = field(default_factory=list)
    stored: bool = False
    duration_ms: float = 0.0
    quarantined: bool = False  # True if pre-remember scan flagged it

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RecallResult:
    """Result of a recall() call, transport-friendly."""
    chunk_id: str = ""
    text: str = ""
    source_path: str = ""
    tier: str = ""
    importance: float = 0.0
    score: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerbatimResult:
    """Result of recall_verbatim() — like RecallResult but `text` is the
    pre-overlap, pre-prefix source bytes (the user's exact words) instead
    of the contextualized chunk as stored. This is the L13 verbatim-first
    contract from MemPalace."""
    chunk_id: str = ""
    verbatim_text: str = ""
    source_path: str = ""
    tier: str = ""
    importance: float = 0.0
    score: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# -----------------------------------------------------------------------------
# The facade
# -----------------------------------------------------------------------------

class Brain:
    """
    Framework-agnostic interface to the DuckBot brain.

    Usage:
        brain = Brain()
        brain.remember("Duckets rotated the bot token today")
        results = brain.recall("bot token rotation", k=5)
        stats = brain.stats()
        for e in brain.graph_query_at("bot token", time.time() - 86400):
            print(e)
        block = brain.block_read("user")
        brain.block_write("today_focus", "Merge brain upgrade PR")

    All methods are sync and return JSON-serializable dataclasses. For async
    access, use the underlying `Memory` class from `src.memory`.
    """

    def __init__(
        self,
        graph_path: Optional[Path] = None,
        blocks_path: Optional[Path] = None,
        quarantine_path: Optional[Path] = None,
        persist_dir: Optional[Path] = None,
        embedder: Any | None = None,
        scan_before_remember: bool = True,
    ):
        self.graph_path = Path(graph_path) if graph_path else DEFAULT_GRAPH_PATH
        self.blocks_path = Path(blocks_path) if blocks_path else DEFAULT_BLOCKS_PATH
        self.quarantine_path = Path(quarantine_path) if quarantine_path else DEFAULT_QUARANTINE_PATH
        self.persist_dir = Path(persist_dir) if persist_dir else None
        self._embedder = embedder
        self.scan_before_remember = scan_before_remember
        self._scanner = InjectionScanner()
        # Lazy-initialized memory (the heavy async one) is created on demand.

    def _memory(self):
        from src.memory import Memory
        if self._embedder is not None:
            return Memory(embedder=self._embedder, persist_dir=self.persist_dir)
        return Memory(persist_dir=self.persist_dir)

    # ------------------------------------------------------------------ remember
    def remember(
        self,
        text: str,
        source_path: str = "<connector>",
        metadata: Optional[dict] = None,
        force_tier: Optional[str] = None,
        skip_scan: bool = False,
        facts: Optional[list[str]] = None,
    ) -> RememberResult:
        """
        Save a memory. Runs the injection scan first unless skip_scan=True.
        If quarantined, returns RememberResult(quarantined=True) and does NOT store.

        `facts` is an optional list of pre-extracted durable facts (strings)
        the host agent already pulled out of `text`. The brain stores each
        as its own semantic-tier chunk with metadata.kind="agent_fact" — no
        extra model load. The agent owns extraction; the brain owns storage.
        """
        from src.tier import Tier, coerce_optional_tier

        # Pre-remember: scan for injection
        if self.scan_before_remember and not skip_scan:
            scan = self._scanner.scan(text)
            if not scan.is_clean:
                # Quarantine it. The user can review and reject (true positive)
                # or approve (false positive) — only approval re-stores it.
                # QuarantineStore.__init__ creates the parent directory, so
                # we can always call it (no need for the pre-guard that the
                # previous version had — that guard was always-true and
                # confused readers).
                with QuarantineStore(path=self.quarantine_path) as q:
                    q.add(scan)
                return RememberResult(
                    quarantined=True,
                    stored=False,
                )

        async def _remember() -> RememberResult:
            mem = self._memory()
            ft = coerce_optional_tier(force_tier)
            r = await mem.remember(
                text,
                source_path=source_path,
                metadata=metadata,
                force_tier=ft,
                facts=facts,
            )
            return RememberResult(
                chunk_id=r.chunk_id,
                tier=r.tier.value if r.tier else None,
                confidence=r.confidence,
                importance=r.importance,
                entities=r.entities,
                relationships=r.relationships,
                stored=r.stored,
                duration_ms=r.duration_ms,
            )

        return _run_async(_remember())

    # ------------------------------------------------------------------- recall
    def recall(
        self,
        query: str,
        k: int = 5,
        tier: Optional[str] = None,
        min_importance: Optional[float] = None,
        rerank: Optional[bool] = None,
        decay: Optional[bool] = None,
        tier_priors: Optional[bool] = None,
        tier_priors_overrides: Optional[dict[str, float]] = None,
        fsrs: Optional[bool] = None,
    ) -> list[RecallResult]:
        """Hybrid retrieval (vector + BM25 + RRF + optional cross-encoder rerank
        + optional Ebbinghaus decay weighting + optional tier priors
        + optional FSRS-6 spaced repetition).

        Args:
            query: search text
            k: top-k results
            tier: restrict to one tier (working/episodic/semantic/procedural)
            min_importance: drop chunks below this importance score
            rerank: True/None forces/enables the cross-encoder pass (Layer 7).
                None reads DUCKBOT_RERANK env var (default off). Pass False to
                explicitly disable for one call.
            decay: True/None forces/enables Ebbinghaus decay (Layer 8). None
                reads DUCKBOT_DECAY env var (default off). Pass False to
                disable. Pure public-domain math, no LLM call.
            tier_priors: True/None forces/enables per-tier prior weighting
                (Layer 11). None reads DUCKBOT_TIER_PRIORS env var (default
                off). Pass False to disable. Defaults: procedural=1.5,
                semantic=1.2, episodic=1.0, working=0.8.
            tier_priors_overrides: Optional dict mapping tier name -> prior
                weight. Tier names not in the dict fall back to defaults.
            fsrs: True/None forces/enables FSRS-6 spaced repetition (Layer 9).
                None reads DUCKBOT_FSRS env var (default off). Pass False to
                disable. Uses per-chunk stability_days + difficulty from
                metadata. Replaces L8 Ebbinghaus retention with FSRS-6
                power-law. Public-domain algorithm spec.
        """
        from src.tier import Tier, coerce_optional_tier

        async def _recall() -> list[RecallResult]:
            mem = self._memory()
            tier_enum = coerce_optional_tier(tier)
            results, _ = await mem.recall(
                query, k=k, tier=tier_enum,
                min_importance=min_importance,
                rerank=rerank, decay=decay,
                tier_priors=tier_priors,
                tier_priors_overrides=tier_priors_overrides,
                fsrs=fsrs,
            )
            out = []
            for r in results:
                meta = r.metadata or {}
                # source_path lives in metadata
                source = meta.get("source_path") or meta.get("source") or "<unknown>"
                importance = float(meta.get("importance", 0.0) or 0.0)
                tier_str = r.tier if isinstance(r.tier, str) else (r.tier.value if r.tier else "")
                out.append(RecallResult(
                    chunk_id=r.chunk_id,
                    text=r.text,
                    source_path=source,
                    tier=tier_str,
                    importance=importance,
                    score=float(r.rrf_score or 0.0),
                    metadata=meta,
                ))
            return out

        return _run_async(_recall())

    # ----------------------------------------------------------------- recall verbatim
    def recall_verbatim(
        self,
        query: str,
        k: int = 5,
        tier: Optional[str] = None,
        min_importance: Optional[float] = None,
        rerank: Optional[bool] = None,
        decay: Optional[bool] = None,
        tier_priors: Optional[bool] = None,
        tier_priors_overrides: Optional[dict[str, float]] = None,
        fsrs: Optional[bool] = None,
    ) -> list[VerbatimResult]:
        """Like `recall()` but returns only the verbatim (pre-overlap) text.

        Implements the L13 verbatim-first storage contract from
        MemPalace's CLAUDE.md (verbatim always). The point: when a user
        asks "what exactly did I say about X?" we return their source bytes,
        not a contextualized chunk with '[...continued from previous
        section...]' prefixes baked in.

        Returns a list of VerbatimResult (a typed dataclass sibling of
        RecallResult). Call .to_dict() on each for the legacy dict shape.
        """
        raw = self.recall(
            query=query, k=k, tier=tier,
            min_importance=min_importance,
            rerank=rerank, decay=decay,
            tier_priors=tier_priors,
            tier_priors_overrides=tier_priors_overrides,
            fsrs=fsrs,
        )
        out: list[VerbatimResult] = []
        for r in raw:
            md = r.metadata or {}
            verbatim = md.get("verbatim_text") or r.text
            out.append(VerbatimResult(
                chunk_id=r.chunk_id,
                verbatim_text=verbatim,
                source_path=r.source_path,
                tier=r.tier,
                importance=r.importance,
                score=r.score,
                metadata=md,
            ))
        return out

    # ----------------------------------------------------------------- wake up
    def wake_up(
        self,
        query: Optional[str] = None,
        k: int = 8,
        include_blocks: bool = True,
        include_graph: bool = True,
        include_fsrs_review: bool = True,
    ) -> dict:
        """One-call "load context for a new session" — MemPalace-inspired.

        Returns a single dict with:
          - `memories`: top-k recalled chunks (filtered to drop superseded ones)
          - `blocks`: active memory blocks (char-bounded, from block_read)
          - `graph_summary`: high-degree entities + recent relationships
          - `fsrs_review_queue`: chunks due for review
          - `stats`: brief store stats (counts per tier)

        Designed for the agent-startup hook path: Hermes agent opens a
        session, calls `brain_wake_up` once, and has everything it needs
        to continue a previous conversation without N round-trips.

        Args:
          query: optional anchor — if provided, recall is run with this
            query; otherwise recall uses the most-recent episodic chunks.
            The query-less path is the "what was I doing recently?" wake-up.
          k: how many memories to recall (default 8).
          include_blocks: include active memory blocks.
          include_graph: include graph summary.
          include_fsrs_review: include FSRS review queue.
        """
        query = (query or "").strip() or None
        out: dict = {"memories": [], "blocks": [], "graph_summary": {},
                     "fsrs_review_queue": [], "stats": {}}

        # Memories — filter out superseded chunks (those with `superseded_by`).
        # Over-fetch aggressively: small k (1-3) was returning 0 results
        # when the top k*2 results were all superseded. Loop with
        # increasing fetch sizes until we have k non-superseded chunks or
        # we've tried hard enough.
        try:
            kept: list = []
            fetch_size = max(k * 5, 20)  # first pass: 5x overshoot
            max_attempts = 5  # cap to bound worst-case latency
            # The fixed query "recent memory" (used when no anchor query
            # is supplied) was being re-embedded on every retry attempt
            # — up to 5 identical embed calls per wake_up. Use the
            # embed cache by setting DUCKBOT_EMBED_CACHE_SIZE (already on
            # by default) and let LMStudioEmbeddings.embed hit it; this
            # drops the 5x redundant HTTP calls to 1.
            for attempt in range(max_attempts):
                if query:
                    raw = self.recall(query=query, k=fetch_size, rerank=True)
                else:
                    raw = self.recall(query="recent memory", k=fetch_size, rerank=False)
                for r in raw:
                    md = r.metadata or {}
                    if md.get("superseded_by"):
                        continue
                    if any(k["chunk_id"] == r.chunk_id for k in kept):
                        continue
                    kept.append(r.to_dict() if hasattr(r, "to_dict") else {
                        "chunk_id": r.chunk_id, "text": r.text, "tier": r.tier,
                        "importance": getattr(r, "importance", 0.0),
                        "score": getattr(r, "score", 0.0),
                        "source_path": getattr(r, "source_path", ""),
                    })
                    if len(kept) >= k:
                        break
                if len(kept) >= k:
                    break
                # Not enough yet — try fetching more.
                fetch_size *= 2

            # v0.15.1: re-rank by 5-factor priority (recency + frequency +
            # connectivity + explicit + type). Recall returned results in
            # RRF order (semantic similarity); priority re-ranking surfaces
            # old-but-important chunks above fresh-but-noisy ones.
            # Falls back gracefully if scoring.py isn't importable.
            try:
                from src.scoring import sort_by_priority
                kept = sort_by_priority(kept)
            except ImportError:
                pass

            out["memories"] = kept
        except Exception as e:
            out["memories_error"] = str(e)

        # Blocks — pull every active block's name + first 200 chars.
        if include_blocks:
            try:
                from src.blocks import BlockStore
                if self.blocks_path.exists():
                    with BlockStore(path=self.blocks_path) as bs:
                        names = bs.names()
                        for n in names[:20]:  # cap at 20 for wake_up speed
                            b = bs.get(n)
                            if b is not None:
                                out["blocks"].append({
                                    "name": b.name,
                                    "preview": b.content[:200],
                                    "char_count": len(b.content),
                                })
            except Exception as e:
                out["blocks_error"] = str(e)

        # Graph summary — top entities by relationship count + recent edges.
        if include_graph:
            try:
                with Graph(path=self.graph_path) as g:
                    # Cheap: just pull counts + recent activity.
                    # Graph.list_entities(kind=None, limit=N) is the right
                    # API — there is no .query() method. Calling the wrong
                    # name silently swallowed the AttributeError into
                    # graph_error and returned empty summaries.
                    ents = g.list_entities(kind=None, limit=20)
                    out["graph_summary"] = {
                        "entity_count": len(ents),
                        "top_entities": [
                            {"name": e.name, "kind": e.kind}
                            for e in ents[:10]
                        ],
                    }
            except Exception as e:
                out["graph_error"] = str(e)

        # FSRS review queue — small (k=5) so we don't blow the context budget.
        if include_fsrs_review:
            try:
                q = self.fsrs_review_queue(k=5)
                out["fsrs_review_queue"] = q
            except Exception as e:
                out["fsrs_review_error"] = str(e)

        # Stats
        try:
            s = self.stats()
            out["stats"] = s if isinstance(s, dict) else getattr(s, "to_dict", lambda: {})()
        except Exception:
            pass

        return out

    # -------------------------------------------------------------- graph: nodes
    def graph_upsert_entity(
        self, name: str, kind: str = "concept", properties: Optional[dict] = None
    ) -> dict:
        """Add or update an entity. Returns the entity as a dict.
        `properties` is stored as `notes` since the underlying schema is
        simple (name/kind/aliases/notes)."""
        if not self.graph_path.exists() and not self.graph_path.parent.exists():
            self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        name = (name or "").strip()
        if not name:
            return {"error": "name is required"}
        kind = (kind or "concept").strip() or "concept"
        try:
            with Graph(path=self.graph_path) as g:
                e = g.upsert_entity(name, kind, aliases=[], notes=str(properties) if properties else None)
                return {
                    "id": e.id,
                    "name": e.name,
                    "kind": e.kind,
                    "aliases": e.aliases,
                    "notes": e.notes,
                }
        except ValueError as exc:
            return {"error": str(exc)}

    def graph_add_relationship(
        self,
        source: str,
        target: str,
        label: str,
        properties: Optional[dict] = None,
    ) -> dict:
        """Add a relationship between two named entities. Creates them if missing."""
        if not self.graph_path.exists() and not self.graph_path.parent.exists():
            self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        source = (source or "").strip()
        target = (target or "").strip()
        label = (label or "").strip()
        if not source or not target or not label:
            return {"error": "source, target, and label are required"}
        try:
            with Graph(path=self.graph_path) as g:
                s = g.upsert_entity(source, "concept")
                t = g.upsert_entity(target, "concept")
                r = g.add_relationship(s.id, t.id, label, source=None)
                return {
                    "id": r.id,
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "label": r.label,
                    "valid_from": r.valid_from,
                    "valid_until": r.valid_until,
                }
        except ValueError as exc:
            return {"error": str(exc)}

    def graph_query(
        self, name: Optional[str] = None, kind: Optional[str] = None, at: Optional[float] = None
    ) -> list[dict]:
        """Query entities. If `name` is given, only matching (case-insensitive substring).
        If `at` is given, only return those active at that time."""
        if not self.graph_path.exists():
            return []
        name = (name or "").strip() or None
        with Graph(path=self.graph_path) as g:
            entities = g.list_entities(kind=kind)
            if name:
                name_l = name.lower()
                entities = [e for e in entities if name_l in e.name.lower()]
            return [e.to_dict() for e in entities]

    # -- Graph Cognify + Reconcile (Cognee ECL stages 2+3) --------------

    def graph_cognify(self, dry_run: bool = True) -> dict:
        """Cognee ECL stage 2: dedupe + reconcile entity relations.

        Walks every relationship, finds pairs (a→b, c→d) where a==c
        AND b==d (same endpoints AND same label) and collapses the
        duplicate. Also merges entity aliases that resolve to the same
        name (case-insensitive).

        Public-domain graph normalization; no LLM call.

        Args:
            dry_run: if True, report what would be merged without
                actually writing. Default True.
        """
        from src.graph import Graph
        if not self.graph_path.exists() and not self.graph_path.parent.exists():
            self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        with Graph(path=self.graph_path) as g:
            dupes = g.find_duplicate_relationships()
            alias_dupes = g.find_duplicate_aliases()
            if not dry_run:
                g.merge_duplicate_relationships(dupes)
                g.merge_duplicate_aliases(alias_dupes)
            return {
                "dry_run": dry_run,
                "duplicate_relationships": len(dupes),
                "duplicate_relationship_samples": [
                    {"source": r[0], "target": r[1], "label": r[2]} for r in dupes[:10]
                ],
                "duplicate_aliases": len(alias_dupes),
                "duplicate_alias_samples": [
                    {"name": a[0], "aliases": a[1]} for a in alias_dupes[:10]
                ],
            }

    def graph_reconcile(self) -> dict:
        """Cognee ECL stage 3: typed-schema reconcile.

        Enforces schema invariants the graph should always have:
          - Every relationship's source/target entity actually exists.
          - Every alias canonicalizes to its entity's name.
          - No self-loops (entity.relationships_to_self) on the same
            label.

        Public-domain graph cleanup; no LLM call.

        Returns: dict with counts of fixes applied. dry_run=False
        always — this method always writes; pass `dry_run=True` via
        `graph_cognify` if you want a preview.
        """
        from src.graph import Graph
        if not self.graph_path.exists() and not self.graph_path.parent.exists():
            self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        with Graph(path=self.graph_path) as g:
            stats = g.reconcile()
        return stats

    def graph_relationships(
        self, entity_name: str, at: Optional[float] = None
    ) -> list[dict]:
        """Get all active relationships for a named entity at time t (default: now)."""
        if not self.graph_path.exists():
            return []
        entity_name = (entity_name or "").strip()
        if not entity_name:
            return []
        with Graph(path=self.graph_path) as g:
            ent = g.find_entity(entity_name)
            if not ent:
                return []
            t = at if at is not None else time.time()
            rels = g.query_active(entity_id=ent.id, at=t)
            return [
                {
                    "id": r.id,
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "label": r.label,
                    "valid_from": r.valid_from,
                    "valid_until": r.valid_until,
                    "is_active": r.is_active,
                }
                for r in rels
            ]

    def graph_history(self, entity_name: str) -> list[dict]:
        """Get the full history (active + ended) of relationships for an entity."""
        if not self.graph_path.exists():
            return []
        entity_name = (entity_name or "").strip()
        if not entity_name:
            return []
        with Graph(path=self.graph_path) as g:
            ent = g.find_entity(entity_name)
            if not ent:
                return []
            rels = g.history(ent.id)
            return [
                {
                    "id": r.id,
                    "source_id": r.source_id,
                    "target_id": r.target_id,
                    "label": r.label,
                    "valid_from": r.valid_from,
                    "valid_until": r.valid_until,
                    "is_active": r.is_active,
                }
                for r in rels
            ]

    # -- Observer: causal precursor tracing + blind-spot detection ---------

    def graph_precursors(
        self,
        entity_name: str,
        *,
        max_depth: int = 3,
        include_inactive: bool = False,
        min_influence: float = 0.0,
    ) -> dict:
        """Backward-trace causal predecessors of an entity.

        Inspired by MindBank's "Observer Perspective". Returns:
          - chain: depth-indexed list of precursors (closest first)
          - influence_modes: top precursors ranked by decayed score
          - critical_depth: shallowest depth capturing >=90% of influence
          - coverage: fraction of immediate edges w/ upstream rationale
          - notes: human-readable diagnostic hints

        Pure delegation to `src/observer.trace_precursors` against the
        live graph. Returns an empty trace (with a note) if the
        graph file doesn't exist or the entity isn't found.
        """
        entity_name = (entity_name or "").strip()
        if not entity_name:
            return {
                "root": "",
                "root_entity_id": "",
                "total_nodes": 0,
                "max_depth_reached": 0,
                "critical_depth": 0,
                "coverage": 0.0,
                "immediate_edge_count": 0,
                "precursors_with_upstream": 0,
                "chain": [],
                "influence_modes": [],
                "notes": ["entity name is required"],
            }
        if not self.graph_path.exists():
            return {
                "root": entity_name, "root_entity_id": "",
                "total_nodes": 0, "max_depth_reached": 0,
                "critical_depth": 0, "coverage": 0.0,
                "immediate_edge_count": 0, "precursors_with_upstream": 0,
                "chain": [], "influence_modes": [],
                "notes": ["graph.db does not exist — no causal trace available"],
            }
        from src.observer import trace_precursors as _trace
        with Graph(path=self.graph_path) as g:
            trace = _trace(
                g, entity_name,
                max_depth=max_depth,
                include_inactive=include_inactive,
                min_influence=min_influence,
            )
            return trace.to_dict()

    def graph_blind_spots(
        self,
        *,
        max_results: int = 50,
        include_inactive: bool = False,
    ) -> list[dict]:
        """Find entities with outgoing causal edges but no upstream rationale.

        Returns a list of BlindSpot dicts sorted by severity
        (high → low) then by causal_edge_count desc. Limited to
        `max_results`.
        """
        if not self.graph_path.exists():
            return []
        from src.observer import find_blind_spots as _find
        with Graph(path=self.graph_path) as g:
            spots = _find(
                g, max_results=max_results, include_inactive=include_inactive,
            )
            return [s.to_dict() for s in spots]

    # -- Inspect: consolidated entity view (Honcho-style) -------------------

    def inspect(self, entity: str, k: int = 10) -> dict:
        """Consolidated entity view: graph + recent memories + blocks.

        Given an entity name (e.g. "Duckets", "OpenClaw", "BATMAN"),
        return everything the brain knows about it in one dict:
          - graph: matching entity + its aliases + active relationships
            + recent history (last 20)
          - memories: top-k recall results mentioning this entity,
            with tier, importance, source, last_recalled_at
          - blocks: any block whose text contains this entity
          - meta: the entity's own kind + recall_count + created_at

        Useful for the agent's audit and self-inspection: "what does the
        brain actually know about X?" without manually joining the graph
        + recall + blocks subsystems.

        Public-domain graph walk + recall; no LLM call.
        """
        entity = (entity or "").strip()
        if not entity:
            return {"error": "entity is required"}
        from src.tier import Tier, coerce_optional_tier
        from src.graph import Graph
        from src.memory import Memory
        from src.blocks import BlockStore
        result: dict = {
            "entity": entity,
            "graph": {"entity": None, "aliases": [], "active_relationships": [],
                       "recent_history": []},
            "memories": [],
            "blocks": [],
            "meta": {},
        }
        # 1. Graph: find the entity + its aliases + active relationships
        if self.graph_path.exists():
            try:
                with Graph(path=self.graph_path) as g:
                    ent = g.find_entity(entity)
                    if ent is not None:
                        result["graph"]["entity"] = {
                            "id": ent.id, "name": ent.name, "kind": ent.kind,
                        }
                        result["graph"]["aliases"] = list(ent.aliases or [])
                        # Active relationships
                        active_rels = g.query_active(
                            entity_id=ent.id, at=None, label=None
                        )
                        result["graph"]["active_relationships"] = [
                            {
                                "id": r.id,
                                "label": r.label,
                                "source_id": r.source_id,
                                "target_id": r.target_id,
                                "valid_from": r.valid_from,
                            }
                            for r in active_rels[:20]
                        ]
                        # Recent history (last 20)
                        history = g.history(ent.id)
                        result["graph"]["recent_history"] = [
                            {
                                "label": r.label,
                                "valid_from": r.valid_from,
                                "valid_until": r.valid_until,
                                "is_active": r.is_active,
                            }
                            for r in history[-20:]
                        ]
                        result["meta"]["kind"] = ent.kind
            except Exception:
                pass
        # 2. Memories: top-k recall mentioning this entity. Use the
        #    existing _run_async bridge to drive the async pipeline from
        #    sync code without leaking event loops.
        try:
            from src.query import hybrid_query
            from src.embeddings import auto_detect_provider
            from src.memory import Memory

            async def _recall():
                mem = self._memory()
                store, _ = await mem._ensure_initialized()
                embedder = await auto_detect_provider()
                # Pass the embedder directly — hybrid_query expects an EmbeddingProvider,
                # not a lambda. The previous lambda hack raised
                # "'function' object has no attribute 'embed_one'" because
                # hybrid_query.embed_one(query_text) was being called on the
                # lambda instead of on the real embedder.
                results, _ = await hybrid_query(
                    entity, store, embedder,
                    n_results=k, tier=None,
                )
                return results

            results = _run_async(_recall())
            for r in results:
                md = r.metadata or {}
                result["memories"].append({
                    "chunk_id": r.chunk_id,
                    "text": (r.text or "")[:300],
                    "tier": r.tier,
                    "importance": float(md.get("importance", 0.0) or 0.0),
                    "source_path": md.get("source_path", ""),
                    "last_recalled_at": float(md.get("last_recalled_at", 0.0) or 0.0),
                    "recall_count": int(md.get("recall_count", 0) or 0),
                })
        except Exception as e:
            result["memories_error"] = str(e)
        # 3. Blocks: scan all blocks for the entity string
        if self.blocks_path.exists():
            try:
                with BlockStore(path=self.blocks_path) as bs:
                    for n in bs.names():
                        b = bs.get(n)
                        if b is not None and entity.lower() in b.content.lower():
                            result["blocks"].append({
                                "name": b.name,
                                "preview": b.content[:300],
                                "char_count": len(b.content),
                                "queued_instructions": b.queued_instructions
                                if hasattr(b, "queued_instructions") else [],
                            })
            except Exception:
                pass
        return result

    # ------------------------------------------------------------- blocks: CRUD
    def block_read(self, name: str) -> Optional[dict]:
        """Read a memory block by name. Returns None if it doesn't exist.

        The returned dict also includes `queued_instructions` — the
        pending entries from `block_rethink()` that an external LLM
        script should drain next. Empty list if no queue or queue empty.
        """
        name = (name or "").strip()
        if not name or not self.blocks_path.exists():
            return None
        with BlockStore(path=self.blocks_path) as s:
            b = s.get(name)
            if b is None:
                return None
            # Surface pending rethink instructions if a queue exists.
            queued = []
            queue_path = self.blocks_path.parent / f"{name}.rethink.jsonl"
            if queue_path.exists():
                import json
                try:
                    with queue_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    queued.append(json.loads(line))
                                except json.JSONDecodeError:
                                    continue
                except OSError:
                    pass
            return {
                "name": b.name,
                "text": b.content,
                "description": b.description,
                "char_limit": b.char_limit,
                "updated_at": b.updated_at,
                "char_count": len(b.content),
                "queued_instructions": queued,
            }

    def block_write(self, name: str, text: str) -> dict:
        """Replace a block's content (creates it if missing)."""
        if not self.blocks_path.exists() and not self.blocks_path.parent.exists():
            self.blocks_path.parent.mkdir(parents=True, exist_ok=True)
        name = (name or "").strip()
        if not name:
            return {"error": "name is required"}
        try:
            with BlockStore(path=self.blocks_path) as s:
                existing = s.get(name)
                if existing is None:
                    b = s.create(name, text)
                    return {"name": name, "action": "created", "updated_at": b.updated_at, "char_count": len(text)}
                b = s.write(name, text)
                return {
                    "name": b.name,
                    "action": "updated",
                    "updated_at": b.updated_at,
                    "char_count": len(text),
                }
        except ValueError as exc:
            return {"error": str(exc)}

    def block_append(self, name: str, text: str) -> dict:
        """Append text to an existing block."""
        name = (name or "").strip()
        if not name:
            return {"error": "name is required"}
        if not self.blocks_path.exists():
            return {"error": f"block not found: {name}"}
        try:
            with BlockStore(path=self.blocks_path) as s:
                s.append(name, text)
                b = s.get(name)
        except (KeyError, ValueError) as e:
            return {"error": str(e)}
        return {"name": b.name, "updated_at": b.updated_at, "char_count": len(b.content)}

    def block_rethink(self, name: str, instruction: str) -> dict:
        """Queue an LLM-driven rethink of a block.

        The brain itself does NOT call any LLM (that's a separate concern).
        Instead, this appends the instruction to the block's "rethink
        queue" — a JSONL file at `data/blocks/<name>.rethink.jsonl`. An
        external LLM script (or the dashboard) drains the queue:

          1. read block via `block_read` (returns queued_instructions)
          2. run the LLM with each queued instruction against the content
          3. write the result via `block_write`
          4. the script clears the queue after applying

        This makes block_rethink a real, durable signal the user can act
        on later — not a silent no-op.

        Args:
            name: block name (created if missing — the rethink queue can
                pre-populate a future block).
            instruction: human-readable prompt to run on the block's content.

        Returns:
            Dict with `queued: True`, `queue_len` (current queue depth),
            and `queue_path` (where the external script should look).
        """
        import json
        import time

        name = (name or "").strip()
        if not name:
            return {"error": "name is required"}
        # Ensure data/blocks/ exists
        self.blocks_path.parent.mkdir(parents=True, exist_ok=True)
        queue_path = self.blocks_path.parent / f"{name}.rethink.jsonl"

        # Append the instruction. JSONL keeps the queue append-only and
        # crash-safe (no partial rewrites if the process dies mid-write).
        entry = {"ts": time.time(), "instruction": instruction}
        with queue_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        # Count the new queue depth.
        queue_len = 0
        if queue_path.exists():
            with queue_path.open("r", encoding="utf-8") as f:
                queue_len = sum(1 for _ in f)

        result = {
            "name": name,
            "instruction": instruction,
            "queued": True,
            "queue_len": queue_len,
            "queue_path": str(queue_path),
            "implemented": True,
            "note": (
                "Queued. An external LLM script drains the queue: read the "
                "block, run each queued instruction, then block_write the "
                "result. See the block_rethink docstring."
            ),
        }
        # Include current block content if it exists, for client convenience.
        b = self.block_read(name)
        if b is not None:
            result["current"] = b
        return result

    def block_delete(self, name: str) -> dict:
        """Delete a memory block by name."""
        name = (name or "").strip()
        if not name:
            return {"deleted": False, "error": "name is required"}
        if not self.blocks_path.exists():
            return {"deleted": False, "reason": "no blocks.db"}
        with BlockStore(path=self.blocks_path) as s:
            ok = s.delete(name)
            return {"deleted": ok, "name": name}

    def block_list(self) -> list[dict]:
        """List all block names + summary."""
        if not self.blocks_path.exists():
            return []
        with BlockStore(path=self.blocks_path) as s:
            return [
                {
                    "name": b.name,
                    "char_count": len(b.content),
                    "updated_at": b.updated_at,
                }
                for b in s.list_blocks()
            ]

    def seed_default_blocks(self) -> list[dict]:
        """Create the 5 default blocks (persona, user, active_project, today_focus, open_questions)."""
        from ..blocks import make_default_blocks
        if not self.blocks_path.exists() and not self.blocks_path.parent.exists():
            self.blocks_path.parent.mkdir(parents=True, exist_ok=True)
        with BlockStore(path=self.blocks_path) as s:
            created = make_default_blocks(s)
        return [{"name": b.name, "char_count": len(b.content), "updated_at": b.updated_at} for b in created]

    # ------------------------------------------------------------- quarantine
    def injection_scan(self, text: str) -> dict:
        """Run a one-shot injection scan, return summary. Does NOT quarantine."""
        r = self._scanner.scan(text)
        return {
            "is_clean": r.is_clean,
            "max_severity": r.max_severity,
            "pattern_hits": [{"id": p.id, "severity": p.severity, "description": p.description} for p, _ in r.pattern_hits],
            "heuristic_hits": [{"name": h.name, "severity": h.severity, "description": h.description} for h in r.heuristic_hits],
            "scan_id": r.scan_id,
        }

    def quarantine_list(self, status: str = "pending") -> list[dict]:
        """List quarantined chunks. status: 'pending', 'approved', 'rejected', or 'all'."""
        if not self.quarantine_path.exists():
            return []
        with QuarantineStore(path=self.quarantine_path) as q:
            if status == "all":
                return q.list_all()
            return q.list_pending() if status == "pending" else q.list_all(status=status)

    def quarantine_review(self, scan_id: str, decision: str, reviewer: str = "operator") -> dict:
        """Approve or reject a quarantined chunk. decision: 'approved' or 'rejected'."""
        scan_id = (scan_id or "").strip()
        decision = (decision or "").strip()
        if not scan_id or not decision:
            return {"error": "scan_id and decision are required"}
        if not self.quarantine_path.exists():
            return {"error": "quarantine not initialized"}
        with QuarantineStore(path=self.quarantine_path) as q:
            ok = q.review(scan_id, decision, reviewer=reviewer)
            return {"scan_id": scan_id, "decision": decision, "ok": ok}

    # -------------------------------------------------------------------- stats
    # New tool surface (v0.10.0 — useful MCP tools extension).
    # Each method is sync + JSON-serializable so it maps 1:1 to an MCP tool
    # exposed by `src.mcp_server.py` AND the OpenClaw extension at
    # `src/extensions/duckbot_brain/adapter.py`.

    # -- FSRS-6 spaced repetition (L9) -----------------------------------
    def fsrs_review_queue(
        self,
        tier: Optional[str] = None,
        k: int = 10,
        now: Optional[float] = None,
    ) -> list[dict]:
        """Return chunks that are due for FSRS-6 review.

        A chunk is "due" when its retrievability R(t, S) drops below
        `REVIEW_THRESHOLD` (default 0.9). Lower R = more urgent review.

        Public-domain math from FSRS-6 algorithm spec. No LLM call.

        Args:
            tier: filter by tier (working/episodic/semantic/procedural)
            k: max number of chunks to inspect (default 10)
            now: current time (default time.time()). Exposed for tests.

        Returns: list of dicts {chunk_id, tier, retrievability, stability_days,
                                difficulty, last_review_ts, urgency}.
        """
        from src.memory import Memory
        from src.tier import Tier, coerce_optional_tier
        from src.fsrs import fsrs_retrievability

        async def _queue() -> list[dict]:
            mem = self._memory()
            tier_enum = coerce_optional_tier(tier)
            t = now if now is not None else time.time()
            # Get a wide net of recent chunks; the FSRS filter is cheap.
            # 100 is enough for the "due for review" view.
            results, _ = await mem.recall(
                "review recent memory",  # dummy query
                k=k * 4,  # oversample to filter
                tier=tier_enum,
            )
            queue = []
            for r in results:
                md = r.metadata or {}
                # FSRS state lives in metadata under these keys (set by L9 path).
                # On a fresh corpus chunks have no FSRS state — use the
                # ingested_at timestamp as a proxy for "first reviewed" so the
                # queue isn't permanently empty. The previous version skipped
                # such chunks entirely (silent no-op on fresh installs).
                stability = float(md.get("fsrs_stability_days") or md.get("stability_days") or 0.0)
                if stability <= 0:
                    stability = 7.0  # default 1-week stability for new chunks
                difficulty = float(md.get("fsrs_difficulty") or md.get("difficulty") or 5.0)
                last_review = float(md.get("fsrs_last_review_ts") or md.get("last_review_ts") or 0.0)
                if last_review <= 0:
                    # No explicit review timestamp → fall back to ingested_at.
                    # If even that is missing, use 0 (chunk appears ancient).
                    last_review = float(md.get("ingested_at") or 0.0)
                elapsed_days = max(0.0, (t - last_review) / 86400.0)
                R = fsrs_retrievability(elapsed_days, stability)
                if R < 0.9:  # REVIEW_THRESHOLD
                    queue.append({
                        "chunk_id": r.chunk_id,
                        "tier": r.tier if isinstance(r.tier, str) else (r.tier.value if r.tier else ""),
                        "retrievability": round(R, 4),
                        "stability_days": round(stability, 2),
                        "difficulty": round(difficulty, 2),
                        "last_review_ts": last_review,
                        "elapsed_days": round(elapsed_days, 2),
                        "urgency": round(1.0 - R, 4),
                        "source_path": r.source_path if hasattr(r, "source_path") else (md.get("source_path") or ""),
                        "preview": (r.text[:120] + "…") if len(r.text) > 120 else r.text,
                    })
            queue.sort(key=lambda x: x["urgency"], reverse=True)
            return queue[:k]

        return _run_async(_queue())

    # -- Ebbinghaus decay status (L8) ------------------------------------
    def decay_status(
        self,
        tier: Optional[str] = None,
        k: int = 50,
        now: Optional[float] = None,
    ) -> dict:
        """Return decay status for a sample of recent chunks.

        For each chunk, computes Ebbinghaus retention R = e^(-t/S) where:
          t = days since the chunk was last "touched" (last review / last remember)
          S = stability days (heuristic: importance * 30)

        Public-domain math (1885). No LLM call.

        Returns: dict with totals + per-tier breakdown + sample chunks.
        """
        from src.memory import Memory
        from src.tier import Tier, coerce_optional_tier
        from src.decay import ebbinghaus_retention

        async def _status() -> dict:
            mem = self._memory()
            tier_enum = coerce_optional_tier(tier)
            t = now if now is not None else time.time()
            results, _ = await mem.recall(
                "recent memory decay status",
                k=k * 2,
                tier=tier_enum,
            )
            by_tier: dict[str, dict] = {}
            sample: list[dict] = []
            total_R = 0.0
            counted = 0
            for r in results:
                md = r.metadata or {}
                importance = float(md.get("importance", 0.5) or 0.5)
                last_touch = float(md.get("last_review_ts") or md.get("created_ts") or md.get("created_at") or 0.0)
                if last_touch <= 0:
                    continue
                tier_str = r.tier if isinstance(r.tier, str) else (r.tier.value if r.tier else "unknown")
                elapsed_days = max(0.0, (t - last_touch) / 86400.0)
                stability_days = max(1.0, importance * 30.0)
                R = ebbinghaus_retention(elapsed_days, stability_days)
                counted += 1
                total_R += R
                bucket = by_tier.setdefault(tier_str, {"count": 0, "avg_retention": 0.0, "decayed_count": 0})
                bucket["count"] += 1
                bucket["avg_retention"] += R
                if R < 0.5:
                    bucket["decayed_count"] += 1
                if len(sample) < 10 and R < 0.7:
                    sample.append({
                        "chunk_id": r.chunk_id,
                        "tier": tier_str,
                        "importance": round(importance, 3),
                        "elapsed_days": round(elapsed_days, 1),
                        "stability_days": round(stability_days, 1),
                        "retention": round(R, 4),
                        "preview": (r.text[:80] + "…") if len(r.text) > 80 else r.text,
                    })
            for b in by_tier.values():
                if b["count"] > 0:
                    b["avg_retention"] = round(b["avg_retention"] / b["count"], 4)
            return {
                "as_of": t,
                "sampled_chunks": counted,
                "avg_retention": round(total_R / counted, 4) if counted else None,
                "by_tier": by_tier,
                "most_decayed_sample": sample,
            }

        return _run_async(_status())

    # -- Decay apply (prune below retention threshold) -------------------

    def decay_apply(
        self,
        tier: Optional[str] = None,
        retention_floor: float = 0.05,
        max_prune: int = 1000,
        dry_run: bool = True,
        now: Optional[float] = None,
    ) -> dict:
        """Prune chunks whose Ebbinghaus retention R has dropped below
        `retention_floor`. Public-domain math (1885); no LLM call.

        This is the "memory decay" cron job — run on a daily schedule to
        keep the episodic tier from growing unbounded. Default dry_run=True
        so the caller can preview; pass dry_run=False to actually delete.

        Args:
            tier: limit to one tier (default: all).
            retention_floor: chunks with R < floor are pruned (default 0.05).
            max_prune: safety cap on chunks deleted in one call.
            dry_run: when True, just report what would be pruned.
            now: current time (default time.time()).

        Returns: dict with {dry_run, tier, retention_floor, scanned,
        would_prune, actually_pruned, ids, most_decayed_sample}.
        """
        from src.tier import Tier, coerce_optional_tier
        from src.memory import Memory
        from src.decay import ebbinghaus_retention

        async def _apply() -> dict:
            mem = self._memory()
            tier_enum = coerce_optional_tier(tier)
            t = now if now is not None else time.time()
            # Walk each tier (overfetch a bit so we don't bias to one tier).
            per_tier: dict[str, int] = {}
            ids_to_prune: list[tuple[str, str]] = []  # (chunk_id, tier)
            sample: list[dict] = []
            for tier_name in ("working", "episodic", "semantic", "procedural"):
                if tier_enum is not None and tier_enum.value != tier_name:
                    continue
                try:
                    coll = mem._store.collection_for(Tier(tier_name))
                    data = coll.get(
                        include=["documents", "metadatas"], limit=10000,
                    )
                except Exception:
                    continue
                ids = (data or {}).get("ids") or []
                docs = (data or {}).get("documents") or []
                metas = (data or {}).get("metadatas") or []
                per_tier[tier_name] = 0
                for cid, doc, md in zip(ids, docs, metas):
                    md = md or {}
                    if md.get("superseded_by"):
                        continue
                    try:
                        S = float(md.get("stability_days") or md.get("importance", 0.0) * 30.0 or 0.0)
                    except (TypeError, ValueError):
                        S = 0.0
                    if S <= 0:
                        S = 0.01
                    last_t = float(md.get("last_recalled_at") or md.get("ingested_at") or t)
                    elapsed_days = max(0.0, (t - last_t) / 86400.0)
                    R = ebbinghaus_retention(elapsed_days, S)
                    if R < retention_floor:
                        if len(ids_to_prune) < max_prune:
                            ids_to_prune.append((cid, tier_name))
                            per_tier[tier_name] += 1
                            if len(sample) < 10:
                                sample.append({
                                    "chunk_id": cid,
                                    "tier": tier_name,
                                    "elapsed_days": round(elapsed_days, 1),
                                    "stability_days": round(S, 2),
                                    "retention": round(R, 4),
                                    "preview": (doc[:80] + "…") if len(doc) > 80 else doc,
                                })

            actually_pruned = 0
            if not dry_run and ids_to_prune:
                for cid, tier_name in ids_to_prune:
                    try:
                        mem._store.collection_for(Tier(tier_name)).delete(
                            ids=[cid],
                        )
                        actually_pruned += 1
                    except Exception:
                        continue
            return {
                "dry_run": dry_run,
                "tier": tier,
                "retention_floor": retention_floor,
                "would_prune": len(ids_to_prune),
                "actually_pruned": actually_pruned,
                "per_tier": per_tier,
                "ids": [cid for cid, _ in ids_to_prune[:50]],  # cap the response
                "most_decayed_sample": sample,
            }
        return _run_async(_apply())

    # -- Forget by query --------------------------------------------------
    def forget_by_query(
        self,
        query: str,
        k: int = 5,
        tier: Optional[str] = None,
    ) -> dict:
        """Forget the top-k chunks matching a query.

        Use case: "I don't want to remember anything about X anymore."
        Different from `brain_forget(chunk_id=...)` which deletes one chunk.

        Returns: {deleted: int, deleted_ids: list[str], results: list}.
        """
        from src.tier import Tier, coerce_optional_tier

        async def _forget() -> dict:
            mem = self._memory()
            tier_enum = coerce_optional_tier(tier)
            results, _ = await mem.recall(query, k=k, tier=tier_enum)
            deleted = []
            for r in results:
                ok = await mem.forget(r.chunk_id)
                if ok:
                    deleted.append(r.chunk_id)
            return {
                "deleted": len(deleted),
                "deleted_ids": deleted,
                "matched": [
                    {
                        "chunk_id": r.chunk_id,
                        "score": float(r.rrf_score or 0.0),
                        "tier": r.tier if isinstance(r.tier, str) else (r.tier.value if r.tier else ""),
                        "source_path": r.source_path,
                        "preview": (r.text[:120] + "…") if len(r.text) > 120 else r.text,
                    }
                    for r in results
                ],
            }

        return _run_async(_forget())

    # -- Verbatim substring search ---------------------------------------
    def search_verbatim(self, needle: str, k: int = 5) -> list[dict]:
        """Exact substring match against the verbatim (pre-overlap) text.

        Different from `recall()` which uses vector + BM25. This is a literal
        string match — useful when you remember a phrase verbatim and want
        the chunk that contains it.

        Returns: list of {chunk_id, verbatim_text, source_path, tier, metadata}.
        """
        async def _search() -> list[dict]:
            mem = self._memory()
            # Wide net — verbatim search is just substring on each chunk's
            # verbatim_text field (stored in metadata per L13).
            results, _ = await mem.recall(
                needle,  # use as the query so we get semantically-related chunks
                k=k * 5,
            )
            out = []
            needle_l = needle.lower()
            seen = set()
            for r in results:
                md = r.metadata or {}
                verbatim = md.get("verbatim_text") or r.text
                if needle_l not in verbatim.lower():
                    continue
                if r.chunk_id in seen:
                    continue
                seen.add(r.chunk_id)
                # Highlight the matches (first 3)
                idx = verbatim.lower().find(needle_l)
                highlights = []
                while idx != -1 and len(highlights) < 3:
                    highlights.append({
                        "start": idx,
                        "end": idx + len(needle),
                        "context": verbatim[max(0, idx - 40): idx + len(needle) + 40],
                    })
                    idx = verbatim.lower().find(needle_l, idx + len(needle))
                out.append({
                    "chunk_id": r.chunk_id,
                    "verbatim_text": verbatim,
                    "source_path": md.get("source_path") or md.get("source") or "",
                    "tier": r.tier if isinstance(r.tier, str) else (r.tier.value if r.tier else ""),
                    "importance": float(md.get("importance", 0.0) or 0.0),
                    "match_count": verbatim.lower().count(needle_l),
                    "highlights": highlights,
                })
                if len(out) >= k:
                    break
            return out

        return _run_async(_search())

    # ----------------------------------------------------- v0.11.0 extensions
    # The next three methods are the v0.11.0 add-on integrations:
    #   - OpenClaw dreaming bridge (DREAMS.md + memory/dreaming/*.md)
    #   - Hermes /learn shim (skill creation + brain ingest)
    #   - Active Memory tool aliases (memory_query / memory_store / etc.)
    # They are exposed on the Brain facade so MCP / OpenClaw / Hermes
    # connectors and CLI callers all share the same entry point.

    def dreaming_read(self) -> dict:
        """Pull DREAMS.md + memory/dreaming/*.md into the brain as `semantic`.

        Idempotent — uses content-hash state to skip already-ingested entries.
        Returns a dict with new_entries, skipped, by_kind, sources.
        """
        from .dreaming import read_dreams
        mem = self._memory()
        return read_dreams(mem)

    def dreaming_cycle(self, k: int = 10, min_importance: float = 0.5) -> dict:
        """Distill high-importance episodic chunks into a new dream entry.

        Writes to memory/dreaming/deep/<date>.md so OpenClaw's dreamer can
        pick it up on its next pass. Returns distilled_chunks, by_tier,
        output_files.
        """
        from .dreaming import write_dream_cycle
        mem = self._memory()
        return write_dream_cycle(mem, k=k, min_importance=min_importance)

    def learn(
        self,
        text: str,
        force_tier: str = "procedural",
        source: str = "<hermes-/learn>",
        metadata: Optional[dict] = None,
        invoke_hermes: bool = True,
    ) -> dict:
        """Hermes /learn shim. Ingest + write to memory/learning/ + invoke
        `hermes learn` if available. Returns chunk_id, written_to,
        hermes_invoked, hermes_output.
        """
        from .learn import learn as _learn
        mem = self._memory()
        return _learn(
            mem,
            text=text,
            force_tier=force_tier,
            source=source,
            metadata=metadata,
            invoke_hermes=invoke_hermes,
        )

    def active_memory(self, tool: str, args: Optional[dict] = None) -> dict:
        """OpenClaw Active Memory tool alias.

        Dispatches `memory_query`, `memory_store`, `memory_recent`,
        `memory_forget` to the brain. Returns {ok, tool, data, error}.
        """
        from .active_memory import make_adapter
        adapter = make_adapter(self)
        return adapter.call(tool, args or {})

    # -------------------------------------------------------------------- stats
    def stats(self, include_vector_store: bool = True) -> BrainStats:
        """One-glance snapshot of all 5 layers.

        `include_vector_store=False` skips the chroma read (useful in tests
        that use a custom graph/blocks/quarantine path but want to avoid
        touching the real chroma).
        """
        s = BrainStats()
        # Vector store
        if include_vector_store:
            try:
                from ..store import MemoryStore
                store = MemoryStore()
                vs = store.stats()
                s.vector_chunks = vs.total
                s.vector_by_tier = {t: getattr(vs, t, 0) for t in ("working", "episodic", "semantic", "procedural")}
            except Exception as e:
                import sys
                print(f"[brain.stats] vector store error: {e}", file=sys.stderr)
        # Graph
        if self.graph_path.exists():
            try:
                with Graph(path=self.graph_path) as g:
                    gs = g.stats()
                    s.graph_entities = gs.get("entities", 0)
                    s.graph_relationships = gs.get("relationships", 0)
                    s.graph_active_relationships = gs.get("active_relationships", 0)
            except Exception:
                pass
        # Blocks
        if self.blocks_path.exists():
            try:
                with BlockStore(path=self.blocks_path) as bs:
                    s.blocks = len(bs.list_blocks())
            except Exception:
                pass
        # Quarantine
        if self.quarantine_path.exists():
            try:
                with QuarantineStore(path=self.quarantine_path) as q:
                    qs = q.stats()
                    s.quarantine_total = qs.get("total", 0)
                    s.quarantine_pending = qs.get("pending", 0)
                    s.quarantine_approved = qs.get("approved", 0)
                    s.quarantine_rejected = qs.get("rejected", 0)
            except Exception:
                pass
        return s
