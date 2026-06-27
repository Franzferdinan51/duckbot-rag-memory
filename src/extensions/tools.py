"""extensions/tools.py — shared "core agent surface" for OpenClaw + Hermes.

Single source of truth for the 12 tools that every thin entry point
(OpenClaw stdio adapter, Hermes MemoryProvider plugin, anything else
that wants zero-deps stdio JSON-RPC / function-call access) should
expose.

Why 12 and not 64 (the full MCP server surface)?

  - These 12 are the ones every skill file (skills/duckbot-brain,
    skills/openclaw-imports, skills/codex-imports, skills/cursor-imports)
    advertises to agents, plus the 2 agent-driven skill-pipeline tools
  - The full 67 are still available via:
      * `python -m src.cli <verb>` (CLI)
      * `scripts/duckbot-ask "..."` (shell wrapper)
      * The canonical MCP server at `src/mcp_server.py` (67 tools)
  - Keeping the thin surface tight means portable stdio JSON-RPC stays
    lightweight and the OpenClaw adapter doesn't have to bundle the
    graph / blocks / quarantine layers (those are admin concerns).

The 12 tools map to existing `src.connectors.base.Brain` methods +
the skill_pipeline module (brain_skills_list / brain_skills_promote).
This module is the dispatch layer: every entry point calls
`dispatch(name, args)` and gets a JSON-serializable dict back.

No LLM, no paid APIs. Local stdlib + the Brain facade only.
"""

# MIT License — see LICENSE in the repository root.


from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from src.connectors.base import Brain, _run_async
from src.tier import Tier, coerce_optional_tier

logger = logging.getLogger(__name__)

_VALID_TIERS = tuple(tier.value for tier in Tier)


def _normalize_tier_arg(args: dict) -> tuple[str | None, dict | None]:
    """Return a canonical optional tier string, plus an error when invalid."""
    tier = args.get("tier")
    try:
        normalized = coerce_optional_tier(tier)
    except ValueError:
        return None, {"error": f"tier must be one of {list(_VALID_TIERS)}, got {tier!r}"}
    return (normalized.value if normalized is not None else None), None


# -----------------------------------------------------------------------------
# Tool schemas (MCP-shaped JSON-Schema, used directly by OpenClaw adapter)
# -----------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "brain_wake_up",
        "description": (
            "ONE-CALL session-startup context load. Returns recent memories "
            "(superseded filtered out), active memory blocks, graph summary, "
            "FSRS review queue, and brief stats — everything an agent needs "
            "to continue a previous conversation without N round-trips. "
            "Use on session start. MemPalace-inspired. With a blank query, "
            "uses the recent-memory wake-up path instead of anchored recall."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "optional anchor; if blank, recent-memory wake-up"},
                "k": {"type": "integer", "default": 8, "description": "max memories to return"},
                "include_blocks": {"type": "boolean", "default": True},
                "include_graph": {"type": "boolean", "default": True},
                "include_fsrs_review": {"type": "boolean", "default": True},
            },
        },
    },
    {
        "name": "brain_recall",
        "description": (
            "Hybrid retrieval (vector + BM25 + RRF). Returns top-k chunks "
            "with tier, source, importance, score. Optional rerank=true for "
            "cross-encoder boost, decay=true for Ebbinghaus retention weighting, "
            "tier_priors=true for per-tier multiplicative weighting (Layer 11), "
            "fsrs=true for FSRS-6 power-law forgetting (Layer 9)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 5},
                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"]},
                "min_importance": {"type": "number"},
                "rerank": {"type": "boolean", "default": False},
                "decay": {"type": "boolean", "default": False},
                "tier_priors": {"type": "boolean", "default": False, "description": "Layer 11: apply per-tier multiplicative weights (procedural=1.5, semantic=1.2, episodic=1.0, working=0.8)"},
                "tier_priors_overrides": {"type": "object", "description": "per-tier weight overrides, e.g. {\"procedural\": 2.0}"},
                "fsrs": {"type": "boolean", "default": False, "description": "Layer 9: use FSRS-6 power-law forgetting instead of Ebbinghaus"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "brain_recall_verbatim",
        "description": (
            "Returns the original (pre-overlap, pre-prefix) source text — "
            "never paraphrased. Use when the user asks 'what exactly did I say?'."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 5},
                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"]},
                "min_importance": {"type": "number"},
                "rerank": {"type": "boolean", "default": False},
                "decay": {"type": "boolean", "default": False},
                "tier_priors": {"type": "boolean", "default": False},
                "tier_priors_overrides": {"type": "object"},
                "fsrs": {"type": "boolean", "default": False},
            },
            "required": ["query"],
        },
    },
    {
        "name": "brain_remember",
        "description": (
            "Persist text to the brain. By default NON-BLOCKING (returns "
            "status=queued, ingest runs on a daemon thread). Rate-limited "
            "(10/min). "
            "Pass kind='skill_candidate' to stamp a lightweight skill "
            "candidate (agent-driven pipeline): the brain just stores + "
            "embeds the chunk in the procedural tier with "
            "metadata.kind='skill_candidate' — NO LLM call. Returns the "
            "chunk_id so the agent can later promote it via "
            "brain_skills_promote. The agent (not the brain) decides if "
            "it's worth a full SKILL.md."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "source": {"type": "string", "description": "where this came from"},
                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"]},
                "kind": {
                    "type": "string",
                    "enum": ["skill_candidate"],
                    "description": "special remember mode. 'skill_candidate' stamps a procedural-tier chunk for later promotion — no LLM, returns chunk_id (blocking).",
                },
                "summary": {"type": "string", "description": "short label for skill candidates (defaults to truncated text)"},
                "importance": {"type": "number", "default": 0.6, "description": "0..1 ranking score for skill candidates"},
                "trust_level": {
                    "type": "string",
                    "enum": ["full", "standard"],
                    "default": "full",
                    "description": "trust_level='full' (default) skips the injection scan since candidates are agent-authored. 'standard' runs the scan and quarantines suspicious content (use if the agent is processing untrusted input).",
                },
                "facts": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "optional list of pre-extracted durable facts the agent already pulled out of `text`. Each is stored as a semantic-tier chunk (metadata.kind='agent_fact') — keeps extraction in the agent's hands (no extra model load by the brain).",
                },
            },
            "required": ["text"],
        },
    },
    {
        "name": "brain_reflect",
        "description": (
            "Sleep-time consolidation: merge episodic chunks into the "
            "semantic tier. Long-running (seconds to minutes for large "
            "brains); call once per cron / day."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "lookback_days": {"type": "integer", "default": 7},
                "max_chunks": {"type": "integer", "default": 200},
            },
        },
    },
    {
        "name": "brain_stats",
        "description": (
            "One-glance snapshot: vector counts per tier, graph entities + "
            "relationships, block count, quarantine totals."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "brain_fsrs_review",
        "description": (
            "Return chunks due for FSRS-6 spaced-repetition review "
            "(R(t,S) < 0.9). Public-domain math, no LLM call."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"]},
                "k": {"type": "integer", "default": 10},
            },
        },
    },
    {
        "name": "brain_decay_status",
        "description": (
            "Return Ebbinghaus decay status (R = e^(-t/S)) for recent chunks, "
            "grouped by tier. Public-domain math (1885), no LLM call."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"]},
                "k": {"type": "integer", "default": 50},
            },
        },
    },
    {
        "name": "brain_search_verbatim",
        "description": (
            "Exact substring match against the verbatim (pre-overlap) text. "
            "Useful when you remember a phrase verbatim."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "needle": {"type": "string"},
                "k": {"type": "integer", "default": 5},
            },
            "required": ["needle"],
        },
    },
    {
        "name": "brain_skills_list",
        "description": (
            "List unpromoted skill-candidate chunks (agent-driven pipeline). "
            "Returns candidates stamped via brain_remember(kind='skill_candidate'), "
            "sorted by recency then importance. The AGENT reads this list and "
            "decides which to promote — the brain does no LLM work. Pass "
            "include_promoted=true to also see already-promoted candidates."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "include_promoted": {"type": "boolean", "default": False},
                "k": {"type": "integer", "default": 50},
            },
        },
    },
    {
        "name": "brain_skills_suggest",
        "description": (
            "Semantic top-N skill candidates matching a query (agent-driven pipeline). "
            "Uses hybrid retrieval (vector + BM25 + RRF) scoped to the procedural "
            "tier, then filters to unpromoted skill candidates. Use this when the "
            "agent is working on a topic and wants to know 'are there candidate "
            "skills about X?' — the agent reads the results and decides which to "
            "promote. No LLM call."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "semantic anchor (e.g. 'docker container restart')"},
                "k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "brain_skills_promote",
        "description": (
            "Promote a skill candidate to a full agentskills.io SKILL.md. "
            "The AGENT authors name/description/instructions using its own "
            "LLM context — the brain is pure storage + template (no LLM). "
            "Writes skills/<slug>/SKILL.md and marks the candidate chunk as "
            "promoted. Pass overwrite=true to re-promote or replace an "
            "existing skill file."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string", "description": "the candidate chunk to promote"},
                "name": {"type": "string", "description": "human-readable skill name (agent-authored)"},
                "description": {"type": "string", "description": "one-line trigger phrase (agent-authored)"},
                "instructions": {"type": "array", "items": {"type": "string"}, "description": "step-by-step (agent-authored) — flat list. For richer SKILL.md bodies (headings, code, tables) use instructions_markdown instead."},
                "instructions_markdown": {"type": "string", "description": "rich markdown body (overrides instructions). Lets the agent author full markdown sections instead of a flat list."},
                "example": {"type": "string", "description": "optional worked example"},
                "emoji": {"type": "string", "description": "optional emoji override"},
                "overwrite": {"type": "boolean", "default": False},
            },
            "required": ["chunk_id", "name", "description"],
        },
    },
]


# Tool-name lookup for fast dispatch.
_TOOL_NAMES: frozenset[str] = frozenset(t["name"] for t in TOOLS)


def tool_names() -> list[str]:
    """Return the canonical tool-name list (first entry = recommended first call)."""
    return [t["name"] for t in TOOLS]


def tool_schemas() -> list[dict]:
    """Return the MCP-shaped tool list (used by OpenClaw adapter's tools/list)."""
    return list(TOOLS)


# -----------------------------------------------------------------------------
# Dispatch
# -----------------------------------------------------------------------------


_BRAIN: Brain | None = None
_BRAIN_LOCK = threading.Lock()


def _get_brain() -> Brain:
    """Lazy Brain singleton — only built on first dispatch."""
    global _BRAIN
    if _BRAIN is None:
        with _BRAIN_LOCK:
            if _BRAIN is None:
                _BRAIN = Brain()
    return _BRAIN


def reset_brain() -> None:
    """Test helper: drop the cached Brain so the next dispatch rebuilds it."""
    global _BRAIN
    _BRAIN = None


def _serialize_recall(results) -> list[dict]:
    """RecallResult -> JSON dict."""
    out: list[dict] = []
    for r in results:
        out.append({
            "chunk_id": getattr(r, "chunk_id", ""),
            "text": getattr(r, "text", ""),
            "tier": getattr(r, "tier", None),
            "importance": float(getattr(r, "importance", 0.0) or 0.0),
            "score": float(getattr(r, "score", 0.0) or 0.0),
            "source_path": getattr(r, "source_path", "") or "",
            "metadata": getattr(r, "metadata", {}) or {},
        })
    return out


def _serialize_stats(s) -> dict:
    """BrainStats -> JSON dict."""
    return {
        "vector_chunks": getattr(s, "vector_chunks", 0),
        "vector_by_tier": dict(getattr(s, "vector_by_tier", {}) or {}),
        "graph_entities": getattr(s, "graph_entities", 0),
        "graph_relationships": getattr(s, "graph_relationships", 0),
        "graph_active_relationships": getattr(s, "graph_active_relationships", 0),
        "blocks": getattr(s, "blocks", 0),
        "quarantine_total": getattr(s, "quarantine_total", 0),
        "quarantine_pending": getattr(s, "quarantine_pending", 0),
        "quarantine_approved": getattr(s, "quarantine_approved", 0),
        "quarantine_rejected": getattr(s, "quarantine_rejected", 0),
        "generated_at": getattr(s, "generated_at", 0.0),
    }


_REMEMBER_POOL = ThreadPoolExecutor(
    max_workers=2,  # bounded so an agent spamming brain_remember can't
                    # stack up 100s of threads hammering LM Studio
    thread_name_prefix="duckbot-remember",
)


def _do_remember_background(brain: Brain, text: str, source: str, facts: Optional[list[str]] = None) -> None:
    """Fire-and-forget remember() — runs on a daemon thread so the
    MCP/JSON-RPC caller doesn't block on embedding + ingest.

    Note: Brain.remember() is SYNCHRONOUS (it wraps the async Memory
    call via _run_async internally). Don't wrap it in asyncio.run() —
    that would call asyncio.run on a non-coroutine and raise
    "a coroutine was expected" (caught here, but the remember was
    silently dropped)."""
    try:
        brain.remember(text=text, source_path=source, facts=facts or None)
    except Exception as e:  # noqa: BLE001
        logger.warning("background remember failed: %s", e)


def _do_reflect(brain: Brain, lookback_days: int, max_chunks: int) -> dict:
    """Run the heavier Memory().reflect() on a worker thread. We can't
    use brain.recall directly because reflect is on Memory, not Brain."""
    from src.memory import Memory
    return _run_async(Memory().reflect(lookback_days=lookback_days, max_chunks=max_chunks))


def dispatch(name: str, args: dict) -> dict:
    """Run one tool call against the Brain facade.

    Returns a JSON-serializable dict. No exceptions escape — handlers
    convert failures to `{"error": "..."}` so the entry point can
    surface them without breaking the protocol envelope.

    For agents that need rate limiting: see `check_rate_limit(name)`
    below — the entry point should call that FIRST and short-circuit
    on `{"error": "rate_limited"}`.
    """
    args = args or {}
    try:
        if name not in _TOOL_NAMES:
            return {"error": f"unknown tool: {name}"}

        brain = _get_brain()

        if name == "brain_wake_up":
            return brain.wake_up(
                query=(args.get("query") or "").strip() or None,
                k=int(args.get("k", 8)),
                include_blocks=bool(args.get("include_blocks", True)),
                include_graph=bool(args.get("include_graph", True)),
                include_fsrs_review=bool(args.get("include_fsrs_review", True)),
            )

        if name == "brain_recall":
            query = (args.get("query") or "").strip()
            if not query:
                return {"error": "query must be a non-empty string"}
            tier, tier_err = _normalize_tier_arg(args)
            if tier_err is not None:
                return tier_err
            # Validate tier_priors_overrides is a dict (or None) before passing through.
            tpo = args.get("tier_priors_overrides")
            if tpo is not None and not isinstance(tpo, dict):
                return {"error": "tier_priors_overrides must be a dict"}
            results = brain.recall(
                query=query,
                k=args.get("k", 5),
                tier=tier,
                min_importance=args.get("min_importance"),
                rerank=bool(args.get("rerank") or False),
                decay=bool(args.get("decay") or False),
                tier_priors=bool(args.get("tier_priors") or False),
                tier_priors_overrides=tpo,
                fsrs=bool(args.get("fsrs") or False),
            )
            return {"results": _serialize_recall(results)}

        if name == "brain_recall_verbatim":
            query = (args.get("query") or "").strip()
            if not query:
                return {"error": "query must be a non-empty string"}
            tier, tier_err = _normalize_tier_arg(args)
            if tier_err is not None:
                return tier_err
            tpo = args.get("tier_priors_overrides")
            if tpo is not None and not isinstance(tpo, dict):
                return {"error": "tier_priors_overrides must be a dict"}
            results = brain.recall_verbatim(
                query=query,
                k=args.get("k", 5),
                tier=tier,
                min_importance=args.get("min_importance"),
                rerank=bool(args.get("rerank") or False),
                decay=bool(args.get("decay") or False),
                tier_priors=bool(args.get("tier_priors") or False),
                tier_priors_overrides=tpo,
                fsrs=bool(args.get("fsrs") or False),
            )
            # Convert VerbatimResult dataclasses to dicts for JSON
            # serialization (otherwise the OpenClaw CLI / MCP client sees
            # the Python repr of the dataclass instead of structured data).
            return {"results": [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in results]}

        if name == "brain_remember":
            text = args["text"]
            source = args.get("source") or "openclaw-extension://ad-hoc"
            kind = args.get("kind")

            # Reject empty / whitespace-only text. Without this, an empty
            # remember() either silently fails in the daemon thread (empty
            # fire-and-forget) or stores a useless empty chunk (skill_candidate
            # path, where the chunk_id becomes the sha256 of "" + source).
            if not text or not text.strip():
                return {"error": "text must be a non-empty string"}

            # Agent-provided pre-extracted facts. Validate shape once here
            # so the background thread doesn't have to.
            facts = args.get("facts")
            if facts is not None and not isinstance(facts, list):
                return {"error": "facts must be a list of strings"}
            if facts:
                facts = [f for f in facts if isinstance(f, str) and f.strip()]

            if kind == "skill_candidate":
                # Agent-driven pipeline: stamp a candidate (blocking, no LLM).
                # Returns chunk_id so the agent can promote it later.
                from src.skill_pipeline import stamp_skill_candidate
                trust_level = args.get("trust_level", "full")
                if trust_level not in ("full", "standard"):
                    return {"error": f"trust_level must be 'full' or 'standard', got {trust_level!r}"}
                result = stamp_skill_candidate(
                    text=text,
                    source=source,
                    summary=args.get("summary", ""),
                    importance=float(args.get("importance", 0.6)),
                    trust_level=trust_level,
                    brain=brain,
                )
                return {
                    "status": "stored",
                    "kind": "skill_candidate",
                    "chunk_id": result.chunk_id,
                    "tier": result.tier,
                    "stored": result.stored,
                }

            # Default: fire-and-forget on a bounded thread pool so the
            # caller doesn't block on embed + ingest. Bounded at 2 workers
            # so a misbehaving agent that hammers brain_remember can't
            # spawn 100s of threads all racing to embed simultaneously
            # (was the root cause of the LM Studio spam in 2026-06-27).
            try:
                _REMEMBER_POOL.submit(_do_remember_background, brain, text, source, facts)
            except RuntimeError:
                # Pool was shut down (process exit) — fall back to inline.
                _do_remember_background(brain, text, source, facts)
            return {"status": "queued", "source": source}

        if name == "brain_reflect":
            return _do_reflect(
                brain,
                lookback_days=int(args.get("lookback_days", 7)),
                max_chunks=int(args.get("max_chunks", 200)),
            )

        if name == "brain_stats":
            return _serialize_stats(brain.stats())

        if name == "brain_fsrs_review":
            tier, tier_err = _normalize_tier_arg(args)
            if tier_err is not None:
                return tier_err
            return {"queue": brain.fsrs_review_queue(
                tier=tier,
                k=int(args.get("k", 10)),
            )}

        if name == "brain_decay_status":
            tier, tier_err = _normalize_tier_arg(args)
            if tier_err is not None:
                return tier_err
            return brain.decay_status(
                tier=tier,
                k=int(args.get("k", 50)),
            )

        if name == "brain_search_verbatim":
            # Reject empty / whitespace-only needle — substring-search on
            # "" trivially matches every chunk (Python's `in` check), and
            # mem.recall() does an expensive semantic search first.
            needle = (args.get("needle") or "").strip()
            if not needle:
                return {"error": "needle must be a non-empty string"}
            return {"matches": brain.search_verbatim(
                needle=needle,
                k=int(args.get("k", 5)),
            )}

        if name == "brain_skills_list":
            from src.skill_pipeline import list_candidates
            return {"candidates": list_candidates(
                brain=brain,
                include_promoted=bool(args.get("include_promoted", False)),
                k=int(args.get("k", 50)),
            )}

        if name == "brain_skills_suggest":
            query = (args.get("query") or "").strip()
            if not query:
                return {"error": "query must be a non-empty string"}
            from src.skill_pipeline import suggest_candidates
            return {"candidates": suggest_candidates(
                brain=brain,
                query=query,
                k=int(args.get("k", 5)),
            )}

        if name == "brain_skills_promote":
            from src.skill_pipeline import promote_candidate
            return promote_candidate(
                chunk_id=args["chunk_id"],
                name=args["name"],
                description=args["description"],
                instructions=args.get("instructions") or [],
                brain=brain,
                example=args.get("example", ""),
                emoji=args.get("emoji"),
                overwrite=bool(args.get("overwrite", False)),
                instructions_markdown=args.get("instructions_markdown"),
            )

        # Unreachable — guarded by the _TOOL_NAMES check above.
        return {"error": f"unknown tool: {name}"}

    except KeyError as e:
        # Missing required arg.
        return {"error": f"missing required argument: {e.args[0] if e.args else e}"}
    except Exception as e:  # noqa: BLE001
        logger.warning("dispatch(%s) failed: %s", name, e)
        return {"error": str(e)}


# -----------------------------------------------------------------------------
# Rate limit (re-exported for entry-point convenience)
# -----------------------------------------------------------------------------


def check_rate_limit(name: str) -> dict | None:
    """Return None if allowed, else a 429-style error dict.

    Entry points (OpenClaw adapter, Hermes plugin) should call this
    BEFORE `dispatch()` and short-circuit when non-None is returned.

    Disabled via `DUCKBOT_RATELIMIT_DISABLE=1`.
    """
    try:
        from src.ratelimit import get_rate_limiter
    except Exception:  # pragma: no cover — ratelimit module absent in some builds
        return None
    allowed, info = get_rate_limiter().check(name)
    if allowed:
        return None
    return {
        "error": "rate_limited",
        "tool": name,
        "limit_per_min": info.get("limit_per_min"),
        "current_tokens": info.get("current_tokens"),
        "retry_after_seconds": info.get("retry_after", 0.0),
        "message": (
            f"Rate limit exceeded for {name} "
            f"({info.get('limit_per_min')}/min). "
            f"Retry in {info.get('retry_after', 0.0)}s. "
            f"Set DUCKBOT_RATELIMIT_DISABLE=1 to disable."
        ),
    }


# -----------------------------------------------------------------------------
# Function-call-shape adapter (for Hermes plugin which speaks OpenAI
# function-call schemas, not MCP inputSchema)
# -----------------------------------------------------------------------------


def function_call_schemas() -> list[dict]:
    """Return the same 12 tools in OpenAI function-call shape.

    Hermes MemoryProvider plugin uses this — it injects schemas into
    the agent's tool list at session start.
    """
    out: list[dict] = []
    for t in TOOLS:
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["inputSchema"],
            },
        })
    return out


def system_prompt_block() -> str:
    """Static text injected into the agent's system prompt.

    Tells the agent which tools it has and the recommended first call.
    Both the OpenClaw extension README and the Hermes plugin pull this
    so they stay in lock-step.
    """
    return (
        "\n\n## Memory tools\n\n"
        "You have access to the DuckBot brain via these tools:\n"
        "- `brain_wake_up` — **call this first on session start**. Returns "
        "recent memories, active blocks, graph summary, FSRS review queue, "
        "and stats in ONE call.\n"
        "- `brain_recall` — hybrid retrieval (vector + BM25 + RRF). "
        "Optionally `rerank=true` for cross-encoder boost, `decay=true` "
        "for Ebbinghaus retention weighting.\n"
        "- `brain_recall_verbatim` — returns the original source bytes, "
        "never paraphrased. Use when the user asks 'what exactly did I say?'.\n"
        "- `brain_remember` — persist a memory. Non-blocking. Pass "
        "kind='skill_candidate' to stamp a lightweight candidate for the "
        "agent-driven skill pipeline (no LLM, returns chunk_id).\n"
        "- `brain_reflect` — sleep-time consolidation of episodic to semantic.\n"
        "- `brain_stats` — one-glance snapshot of the brain.\n"
        "- `brain_fsrs_review` — chunks due for spaced-repetition review.\n"
        "- `brain_decay_status` — retention scoring for recent chunks.\n"
        "- `brain_search_verbatim` — exact substring match.\n"
        "- `brain_skills_list` — list unpromoted skill candidates. The agent "
        "reads these and decides which to promote.\n"
        "- `brain_skills_promote` — promote a candidate to a full SKILL.md. "
        "The agent authors the content; the brain is pure template.\n\n"
        "**Agent-driven skill pipeline (zero VRAM on brain side):**\n"
        "1. Finish a task worth repeating → `brain_remember(kind='skill_candidate', ...)`\n"
        "2. At a quiet moment → `brain_skills_list` to review candidates\n"
        "3. Write the SKILL.md content yourself (you have the full context)\n"
        "4. `brain_skills_promote(chunk_id=..., name=..., description=..., instructions=[...])`\n"
        "The brain never calls an LLM — only the embedding model runs.\n\n"
        "Proactively search memory before answering questions about prior "
        "work, decisions, or preferences. Memory is local and free to query.\n"
    )


# -----------------------------------------------------------------------------
# One-line CLI summary (for `python -c "from src.extensions.tools import ..."`)
# -----------------------------------------------------------------------------


def summary() -> str:
    """Human-readable one-liner of the surface — for logs and docs."""
    return f"duckbot-rag-memory core agent surface: {len(TOOLS)} tools: {', '.join(tool_names())}"


__all__ = [
    "TOOLS",
    "tool_names",
    "tool_schemas",
    "function_call_schemas",
    "dispatch",
    "check_rate_limit",
    "system_prompt_block",
    "summary",
    "reset_brain",
]
