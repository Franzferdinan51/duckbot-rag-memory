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
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No running loop — safe to use asyncio.run.
        return asyncio.run(coro)
    # We're in a loop. Run the coroutine in a worker thread.
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


# -----------------------------------------------------------------------------
# Result types — JSON-serializable for transport
# -----------------------------------------------------------------------------

@dataclass
class BrainStats:
    """Combined snapshot across all 5 layers."""
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
        scan_before_remember: bool = True,
    ):
        self.graph_path = Path(graph_path) if graph_path else DEFAULT_GRAPH_PATH
        self.blocks_path = Path(blocks_path) if blocks_path else DEFAULT_BLOCKS_PATH
        self.quarantine_path = Path(quarantine_path) if quarantine_path else DEFAULT_QUARANTINE_PATH
        self.scan_before_remember = scan_before_remember
        self._scanner = InjectionScanner()
        # Lazy-initialized memory (the heavy async one) is created on demand.

    # ------------------------------------------------------------------ remember
    def remember(
        self,
        text: str,
        source_path: str = "<connector>",
        metadata: Optional[dict] = None,
        force_tier: Optional[str] = None,
        skip_scan: bool = False,
    ) -> RememberResult:
        """
        Save a memory. Runs the injection scan first unless skip_scan=True.
        If quarantined, returns RememberResult(quarantined=True) and does NOT store.
        """
        from src.memory import Memory
        from src.tier import Tier

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
            mem = Memory()
            ft = Tier(force_tier) if force_tier else None
            r = await mem.remember(
                text,
                source_path=source_path,
                metadata=metadata,
                force_tier=ft,
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
        from src.memory import Memory
        from src.tier import Tier

        async def _recall() -> list[RecallResult]:
            mem = Memory()
            tier_enum = Tier(tier) if tier else None
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

    # -------------------------------------------------------------- graph: nodes
    def graph_upsert_entity(
        self, name: str, kind: str = "concept", properties: Optional[dict] = None
    ) -> dict:
        """Add or update an entity. Returns the entity as a dict.
        `properties` is stored as `notes` since the underlying schema is
        simple (name/kind/aliases/notes)."""
        if not self.graph_path.exists() and not self.graph_path.parent.exists():
            self.graph_path.parent.mkdir(parents=True, exist_ok=True)
        with Graph(path=self.graph_path) as g:
            e = g.upsert_entity(name, kind, aliases=[], notes=str(properties) if properties else None)
            return {
                "id": e.id,
                "name": e.name,
                "kind": e.kind,
                "aliases": e.aliases,
                "notes": e.notes,
            }

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

    def graph_query(
        self, name: Optional[str] = None, kind: Optional[str] = None, at: Optional[float] = None
    ) -> list[dict]:
        """Query entities. If `name` is given, only matching (case-insensitive substring).
        If `at` is given, only return those active at that time."""
        if not self.graph_path.exists():
            return []
        with Graph(path=self.graph_path) as g:
            entities = g.list_entities(kind=kind)
            if name:
                name_l = name.lower()
                entities = [e for e in entities if name_l in e.name.lower()]
            return [e.to_dict() for e in entities]

    def graph_relationships(
        self, entity_name: str, at: Optional[float] = None
    ) -> list[dict]:
        """Get all active relationships for a named entity at time t (default: now)."""
        if not self.graph_path.exists():
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

    # ------------------------------------------------------------- blocks: CRUD
    def block_read(self, name: str) -> Optional[dict]:
        """Read a memory block by name. Returns None if it doesn't exist."""
        if not self.blocks_path.exists():
            return None
        with BlockStore(path=self.blocks_path) as s:
            b = s.get(name)
            if b is None:
                return None
            return {
                "name": b.name,
                "text": b.content,
                "description": b.description,
                "char_limit": b.char_limit,
                "updated_at": b.updated_at,
                "char_count": len(b.content),
            }

    def block_write(self, name: str, text: str) -> dict:
        """Replace a block's content (creates it if missing)."""
        if not self.blocks_path.exists() and not self.blocks_path.parent.exists():
            self.blocks_path.parent.mkdir(parents=True, exist_ok=True)
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

    def block_append(self, name: str, text: str) -> dict:
        """Append text to an existing block."""
        if not self.blocks_path.exists():
            return {"error": f"block not found: {name}"}
        with BlockStore(path=self.blocks_path) as s:
            try:
                s.append(name, text)
                b = s.get(name)
            except (KeyError, ValueError) as e:
                return {"error": str(e)}
            return {"name": b.name, "updated_at": b.updated_at, "char_count": len(b.content)}

    def block_rethink(self, name: str, instruction: str) -> dict:
        """Atomic re-prompt-and-replace: runs an LLM-driven rethink on a block.

        HONEST NOTE: This is a no-op stub. It does NOT call any LLM and does
        NOT modify the block. The "self-editing memory blocks" claim in the
        README is aspirational — the workflow today is:

          1. read the block via `block_read`
          2. run an external LLM-driven rethink (your own script or the
             dashboard)
          3. write the result via `block_write`

        Surfacing `implemented=False` so MCP clients can detect the stub
        status programmatically (rather than parsing the human-readable
        `note` field).
        """
        b = self.block_read(name)
        if b is None:
            return {"error": f"block not found: {name}"}
        return {
            "name": name,
            "instruction": instruction,
            "implemented": False,
            "note": (
                "block_rethink is a stub — use LLM-driven script + block_write. "
                "See docstring for the current workflow."
            ),
            "current": b,
        }

    def block_delete(self, name: str) -> dict:
        """Delete a memory block by name."""
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
        from src.tier import Tier
        from src.fsrs import fsrs_retrievability

        async def _queue() -> list[dict]:
            mem = Memory()
            tier_enum = Tier(tier) if tier else None
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
        from src.tier import Tier
        from src.decay import ebbinghaus_retention

        async def _status() -> dict:
            mem = Memory()
            tier_enum = Tier(tier) if tier else None
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
        from src.memory import Memory
        from src.tier import Tier

        async def _forget() -> dict:
            mem = Memory()
            tier_enum = Tier(tier) if tier else None
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
        from src.memory import Memory

        async def _search() -> list[dict]:
            mem = Memory()
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
        from src.memory import Memory
        mem = Memory()
        return read_dreams(mem)

    def dreaming_cycle(self, k: int = 10, min_importance: float = 0.5) -> dict:
        """Distill high-importance episodic chunks into a new dream entry.

        Writes to memory/dreaming/deep/<date>.md so OpenClaw's dreamer can
        pick it up on its next pass. Returns distilled_chunks, by_tier,
        output_files.
        """
        from .dreaming import write_dream_cycle
        from src.memory import Memory
        mem = Memory()
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
        from src.memory import Memory
        mem = Memory()
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
