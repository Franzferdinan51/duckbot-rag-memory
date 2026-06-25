"""
mcp_server.py - Model Context Protocol server for DuckBot memory.

Exposes the Memory facade as MCP tools so any MCP-aware client (OpenClaw
agents, Claude Code, Cursor, Codex, etc.) can remember/recall directly.

Tools:
  - remember(text, source_path?, metadata?)  → store a memory
  - recall(query, k?, tier?)                  → hybrid retrieval
  - reflect()                                 → consolidate episodic
  - forget(chunk_id)                          → delete a memory
  - stats()                                   → dashboard snapshot
  - watch(paths?)                             → start the auto-update daemon

Run: `python -m src.mcp_server` (stdio) or `python -m src.mcp_server --http PORT`
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add parent to path so this can be run as `python -m src.mcp_server`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory import Memory
from src.tier import Tier


# -----------------------------------------------------------------------------
# Tool definitions (MCP format)
# -----------------------------------------------------------------------------

TOOLS = [
    {
        "name": "remember",
        "description": "Save a memory. Auto-chunks, classifies tier, extracts entities, embeds, stores. Returns chunk_id, tier, importance.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "the memory content"},
                "source_path": {"type": "string", "description": "where this came from (e.g. file path, conversation id)", "default": "<remember>"},
                "metadata": {"type": "object", "description": "arbitrary metadata to attach"},
                "force_tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"], "description": "override auto tier"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "recall",
        "description": "Hybrid retrieval over all chunks. Returns top-k with tier, source, importance, score.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "the search query"},
                "k": {"type": "integer", "default": 5, "description": "number of results"},
                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"], "description": "filter by tier"},
                "min_importance": {"type": "number", "description": "filter by importance threshold (0..1)"},
                "rerank": {"type": "boolean", "default": False, "description": "Layer 7: run cross-encoder rerank with BAAI/bge-reranker-base (MIT, local). No paid API. Off by default; pass true to opt in."},
                "decay": {"type": "boolean", "default": False, "description": "Layer 8: apply Ebbinghaus retention weighting. Public-domain math (1885), no LLM call. Off by default; pass true to opt in."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "reflect",
        "description": "Consolidate episodic chunks into semantic memory. Returns summary of what was merged.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "lookback_days": {"type": "integer", "default": 7},
                "max_chunks": {"type": "integer", "default": 200},
            },
        },
    },
    {
        "name": "forget",
        "description": "Delete a memory by chunk_id.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string"},
                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"]},
            },
            "required": ["chunk_id"],
        },
    },
    {
        "name": "stats",
        "description": "One-glance snapshot of brain state: chunks by tier, last activity, LM Studio reachability.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "watch",
        "description": "Start or stop the file-watcher daemon that auto-ingests new files into the brain.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "paths": {"type": "array", "items": {"type": "string"}, "description": "paths to watch"},
                "daemon": {"type": "boolean", "default": True, "description": "fork into background"},
            },
        },
    },
    {
        "name": "doctor",
        "description": "Run health checks: Python version, critical deps, embedder status, vector store health.",
        "inputSchema": {"type": "object", "properties": {}},
    },

    # ---- v0.10.0 — useful MCP tools extension ----
    {
        "name": "recall_verbatim",
        "description": "Verbatim-first recall. Returns chunks whose exact text was used, with surrounding context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 5},
                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"]},
                "rerank": {"type": "boolean", "default": False},
                "decay": {"type": "boolean", "default": False},
            },
            "required": ["query"],
        },
    },
    {
        "name": "fsrs_review",
        "description": "Get the FSRS-6 spaced-repetition review queue. Items due for review first.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"]},
                "k": {"type": "integer", "default": 10},
            },
        },
    },
    {
        "name": "decay_status",
        "description": "Show retention/decay status for the most recent chunks. Highlights items close to forgetting.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"]},
                "k": {"type": "integer", "default": 50},
            },
        },
    },
    {
        "name": "forget_by_query",
        "description": "Delete the top-k chunks matching a query. Use carefully — destructive.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 5},
                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"]},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_verbatim",
        "description": "Exact substring match against stored verbatim text. Fast and precise.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "needle": {"type": "string"},
                "k": {"type": "integer", "default": 5},
            },
            "required": ["needle"],
        },
    },
    # ---- v0.11.0 — OpenClaw dreaming bridge + Hermes /learn + Active Memory ----
    {
        "name": "dreaming_read",
        "description": "Pull dream entries from OpenClaw dreaming surface into the brain. Idempotent.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "dreaming_cycle",
        "description": "Distill high-importance chunks into a new dream entry.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "k": {"type": "integer", "default": 10, "description": "max chunks to consider"},
                "min_importance": {"type": "number", "default": 0.5, "description": "floor for inclusion"},
            },
        },
    },
    {
        "name": "learn",
        "description": "Ingest a reusable rule into the brain and (optionally) invoke Hermes /learn.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "force_tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"], "default": "procedural"},
                "source": {"type": "string", "default": "<hermes-/learn>"},
                "metadata": {"type": "object"},
                "invoke_hermes": {"type": "boolean", "default": True},
            },
            "required": ["text"],
        },
    },
    {
        "name": "active_memory",
        "description": "Dispatch an Active Memory tool call (memory_query, memory_store, memory_recent, memory_forget).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": ["memory_query", "memory_store", "memory_recent", "memory_forget"]},
                "args": {"type": "object", "description": "args for the inner tool"},
            },
            "required": ["tool"],
        },
    },
    # ---- v0.11.2 — Enhanced Brain: context inflation ----
    {
        "name": "brain_inflate",
        "description": "Enhanced brain inflation. Recall relevant memories and format them for direct injection into agent context. Use when an agent starts a new task, session, or asks 'what do I know about X?' Returns a ready-to-paste markdown block with tier labels, importance scores, and source attribution.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "what the agent is currently working on or asking about"},
                "k": {"type": "integer", "default": 10, "description": "max memories to return"},
                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"], "description": "filter by tier"},
                "min_importance": {"type": "number", "default": 0.3, "description": "minimum importance threshold (0..1)"},
                "agent_name": {"type": "string", "description": "agent name to personalize context (e.g. 'mavis', 'duckbot')"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "brain_sync",
        "description": "Sync stored memories back to agent context files. Writes to OpenClaw (~/.openclaw/workspace/memory/), Hermes (~/.hermes/memories/), or both. Call this after ingest or on a cron to keep the enhanced brain's context files fresh — agents read them on startup so they don't start from a blank slate.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "enum": ["openclaw", "hermes", "both"], "default": "openclaw", "description": "which agent's memory files to write"},
                "memory_k": {"type": "integer", "default": 20, "description": "max memories per tier for MEMORY.md"},
                "user_k": {"type": "integer", "default": 15, "description": "max facts for USER.md"},
                "dry_run": {"type": "boolean", "default": False, "description": "preview what would be written without writing files"},
            },
        },
    },
    {
        "name": "brain_wake_up",
        "description": "One-call session-startup context load. Returns top-k recent memories (filtered to drop superseded ones), active memory blocks, a graph summary, the FSRS review queue, and brief stats — everything an agent needs to continue a previous conversation without N round-trips. Use on session start (MemPalace-inspired wake-up command). Query-less mode returns the most-recent episodic + procedural chunks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "optional anchor — if provided, recall runs with this query; otherwise pulls most-recent episodic chunks"},
                "k": {"type": "integer", "default": 8, "description": "max memories to return"},
                "include_blocks": {"type": "boolean", "default": True, "description": "include active memory blocks"},
                "include_graph": {"type": "boolean", "default": True, "description": "include graph summary (top entities)"},
                "include_fsrs_review": {"type": "boolean", "default": True, "description": "include FSRS review queue"},
            },
        },
    },
    {
        "name": "brain_index",
        "description": "Compact one-line-per-chunk summary of the whole brain. Uses the AAAK compression dialect — an LLM can scan thousands of entries in <500 tokens and then call brain_recall with the source_path of entries it wants to expand. Default 5000 chunks; pair with --tier to narrow.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"], "description": "filter to one tier"},
                "max_chunks": {"type": "integer", "default": 5000, "description": "cap on lines emitted"},
                "preview_chars": {"type": "integer", "default": 80, "description": "chars of each chunk's text to include"},
            },
        },
    },
    {
        "name": "brain_nudge",
        "description": "Proactive memory nudge: surfaces stale-but-important memories the agent might be forgetting about. Logic: high importance + not recently recalled + older than 7 days. Use on a cron or as a 'just-in-time' reminder mid-task. Optional --context to bias toward memories relevant to a current focus.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "context": {"type": "string", "description": "optional current focus — biases nudge toward relevant memories"},
                "k": {"type": "integer", "default": 5, "description": "max memories to return"},
                "min_importance": {"type": "number", "default": 0.6, "description": "importance threshold (0..1)"},
                "stale_days": {"type": "integer", "default": 7, "description": "consider stale if last_recalled_at is older than this many days"},
            },
        },
    },
    {
        "name": "brain_skill_create",
        "description": "Auto-generate an agentskills.io-compatible SKILL.md from a task description + instructions. Writes to skills/<slug>/SKILL.md so the agent can re-use the procedure on similar tasks. Use after a successful task to make the win repeatable.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "human-readable skill name (e.g. 'Restart the BATMAN container')"},
                "description": {"type": "string", "description": "one-line: when to use this skill"},
                "instructions": {"type": "array", "items": {"type": "string"}, "description": "step-by-step instructions"},
                "example": {"type": "string", "description": "optional worked example"},
                "emoji": {"type": "string", "description": "optional emoji override; default is keyword-guessed"},
                "overwrite": {"type": "boolean", "default": False, "description": "replace an existing skill with the same slug"},
            },
            "required": ["name", "description", "instructions"],
        },
    },
    {
        "name": "brain_user_model",
        "description": "Aggregate user-related facts into a single 'user' memory block. Honcho-inspired: a continuously-updated user model that agents can read on session start. Pulls high-importance facts about the user (entities, preferences, recurring context) and writes them to the 'user' block via block_write. Call on a cron or after major life/work changes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "block_name": {"type": "string", "default": "user", "description": "which block to write the model to"},
                "min_importance": {"type": "number", "default": 0.5, "description": "importance threshold (0..1)"},
                "max_facts": {"type": "integer", "default": 30, "description": "max facts to include in the model"},
                "k_per_query": {"type": "integer", "default": 50, "description": "candidates to scan per tier"},
            },
        },
    },
    {
        "name": "brain_palace",
        "description": "Wing/Room/Drawer 2D view of the brain (MemPalace-inspired). With no args, lists every wing (person/project) and its room count. With --wing, returns the drawers in that wing, optionally filtered to one room and/or tier. Use this for project-scoped recall ('everything I know about OpenClaw from this week') without manually filtering by source_path.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "wing": {"type": "string", "description": "walk a specific wing (person/project)"},
                "room": {"type": "string", "description": "filter to one room (date or filename)"},
                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"], "description": "filter to one tier"},
                "max_drawers": {"type": "integer", "default": 100, "description": "cap on drawers returned"},
            },
        },
    },
]


# -----------------------------------------------------------------------------
# Connector tools (Layers 1-4 + dashboard) - backed by the framework-agnostic
# Brain facade. OpenClaw picks them up automatically because they're in TOOLS.
# -----------------------------------------------------------------------------

# Tool definitions and handlers are loaded lazily so a missing layer doesn't
# break the whole server. We populate them right after HANDLERS is defined below.
def _register_connector_tools() -> None:
    try:
        from src.connectors.openclaw import TOOL_DEFINITIONS, handle as _handle
        TOOLS.extend(TOOL_DEFINITIONS)
        for t in TOOL_DEFINITIONS:
            HANDLERS[t["name"]] = (lambda args, h=_handle, n=t["name"]: h(n, args))
    except Exception as _e:
        import sys as _s
        print(f"[mcp_server] connector tools not loaded: {_e}", file=_s.stderr)


# -----------------------------------------------------------------------------
# Tool implementations
# -----------------------------------------------------------------------------

async def handle_remember(args: dict) -> dict:
    mem = Memory()
    from src.tier import Tier
    ft = args.get("force_tier")
    force = Tier(ft) if ft else None
    r = await mem.remember(
        args["text"],
        source_path=args.get("source_path", "<remember>"),
        metadata=args.get("metadata"),
        force_tier=force,
    )
    return {
        "chunk_id": r.chunk_id,
        "tier": r.tier.value,
        "confidence": r.confidence,
        "importance": r.importance,
        "entities": r.entities,
        "relationships": r.relationships,
        "stored": r.stored,
        "duration_ms": r.duration_ms,
    }


async def handle_recall(args: dict) -> dict:
    mem = Memory()
    results, stats = await mem.recall(
        args["query"],
        k=args.get("k", 5),
        tier=args.get("tier"),
        min_importance=args.get("min_importance"),
        rerank=args.get("rerank"),
        decay=args.get("decay"),
    )
    return {
        "results": [r.to_dict() for r in results],
        "stats": stats.to_dict(),
    }


async def handle_reflect(args: dict) -> dict:
    mem = Memory()
    return await mem.reflect(
        lookback_days=args.get("lookback_days", 7),
        max_chunks=args.get("max_chunks", 200),
    )


async def handle_forget(args: dict) -> dict:
    mem = Memory()
    from src.tier import Tier
    tier = Tier(args["tier"]) if args.get("tier") else None
    ok = await mem.forget(args["chunk_id"], tier=tier)
    return {"deleted": ok}


# ---- v0.10.0 — useful MCP tools extension ----

async def handle_recall_verbatim(args: dict) -> dict:
    from src.connectors.base import Brain
    brain = Brain()
    return {"results": brain.recall_verbatim(
        query=args["query"],
        k=args.get("k", 5),
        tier=args.get("tier"),
        rerank=args.get("rerank"),
        decay=args.get("decay"),
    )}


async def handle_fsrs_review(args: dict) -> dict:
    from src.connectors.base import Brain
    brain = Brain()
    return {"queue": brain.fsrs_review_queue(
        tier=args.get("tier"), k=args.get("k", 10)
    )}


async def handle_decay_status(args: dict) -> dict:
    from src.connectors.base import Brain
    brain = Brain()
    return brain.decay_status(tier=args.get("tier"), k=args.get("k", 50))


async def handle_forget_by_query(args: dict) -> dict:
    from src.connectors.base import Brain
    brain = Brain()
    return brain.forget_by_query(
        query=args["query"],
        k=args.get("k", 5),
        tier=args.get("tier"),
    )


async def handle_search_verbatim(args: dict) -> dict:
    from src.connectors.base import Brain
    brain = Brain()
    return {"matches": brain.search_verbatim(
        needle=args["needle"], k=args.get("k", 5),
    )}


# ---- v0.11.0 — OpenClaw dreaming bridge + Hermes /learn + Active Memory ----

async def handle_dreaming_read(args: dict) -> dict:
    from src.connectors.base import Brain
    brain = Brain()
    return brain.dreaming_read()


async def handle_dreaming_cycle(args: dict) -> dict:
    from src.connectors.base import Brain
    brain = Brain()
    return brain.dreaming_cycle(
        k=args.get("k", 10),
        min_importance=args.get("min_importance", 0.5),
    )


async def handle_learn(args: dict) -> dict:
    from src.connectors.base import Brain
    brain = Brain()
    return brain.learn(
        text=args["text"],
        force_tier=args.get("force_tier", "procedural"),
        source=args.get("source", "<hermes-/learn>"),
        metadata=args.get("metadata"),
        invoke_hermes=args.get("invoke_hermes", True),
    )


async def handle_active_memory(args: dict) -> dict:
    from src.connectors.base import Brain
    brain = Brain()
    return brain.active_memory(
        tool=args["tool"],
        args=args.get("args", {}),
    )


async def handle_stats(args: dict) -> dict:
    mem = Memory()
    snap = await mem.stats()
    return {
        "total": snap.total,
        "by_tier": snap.by_tier,
        "by_provider": snap.by_provider,
        "last_remember_ts": snap.last_remember_ts,
        "last_recall_ts": snap.last_recall_ts,
        "lmstudio_reachable": snap.lmstudio_reachable,
    }


async def handle_watch(args: dict) -> dict:
    """Spawn the watcher (daemon by default). Returns PID."""
    paths = args.get("paths") or None
    daemon = args.get("daemon", True)
    # Subprocess so we don't block the MCP server
    import subprocess
    cmd = [sys.executable, "-m", "src.watcher"]
    if daemon:
        cmd.append("daemon")
    else:
        cmd.append("run")
    if paths:
        cmd.extend(paths)
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return {"pid": proc.pid, "command": " ".join(cmd)}


async def handle_doctor(args: dict) -> dict:
    """Same as CLI doctor but as a tool."""
    import importlib
    checks = []
    # Python
    import sys as _s
    checks.append({"name": "python", "value": f"{_s.version_info.major}.{_s.version_info.minor}.{_s.version_info.micro}", "ok": True})
    # Critical deps
    for mod in ["chromadb", "httpx", "numpy"]:
        try:
            importlib.import_module(mod)
            checks.append({"name": mod, "value": "imported", "ok": True})
        except ImportError as exc:
            checks.append({"name": mod, "value": str(exc), "ok": False})
    # Embedder
    mem = Memory()
    try:
        store, emb = await mem._ensure_initialized()
        checks.append({"name": "embedder", "value": f"{emb.name} ({emb.dim}d)", "ok": True})
    except Exception as exc:
        checks.append({"name": "embedder", "value": str(exc), "ok": False})
    # LM Studio
    from src.embeddings import is_lmstudio_reachable
    ok = await is_lmstudio_reachable()
    checks.append({"name": "lmstudio", "value": "reachable" if ok else "unreachable", "ok": ok})
    return {"checks": checks}


# -----------------------------------------------------------------------------
# v0.11.2 — Enhanced Brain: context inflation
# The brain that enhances agents, not just stores memories.
# -----------------------------------------------------------------------------

async def handle_brain_inflate(args: dict) -> dict:
    """Recall relevant memories and format them for direct agent-context injection.

    This is the core "enhanced brain" operation: instead of just storing and
    retrieving, this FEEDS memories back into agent context in a form that's
    immediately useful — markdown with tier labels, importance scores, sources.

    Use when:
      - An agent starts a new session or task
      - An agent asks "what do I know about X?"
      - After ingest to surface what was just learned
    """
    mem = Memory()
    query = args["query"]
    k = args.get("k", 10)
    tier_filter = args.get("tier")
    min_imp = args.get("min_importance", 0.3)
    agent_name = args.get("agent_name", "agent")

    results, _ = await mem.recall(
        query, k=k * 3,  # over-fetch, filter below
        tier=tier_filter,
        min_importance=min_imp,
    )

    # Filter to top-k and group by tier. The previous version broke the
    # inner loop as soon as every tier had k/4+1 items — but the `all()`
    # predicate only fires AFTER all four tiers crossed the threshold, so
    # any tier that never crossed it (e.g. an empty tier) would silently
    # leave the loop with the other tiers under-filled. The fix iterates
    # the full over-fetched candidate set so each tier gets a fair shot.
    by_tier: dict[str, list] = {t.value: [] for t in [Tier.WORKING, Tier.EPISODIC, Tier.SEMANTIC, Tier.PROCEDURAL]}
    for r in results[:k * 4]:  # over-fetch so each tier can fill k/4
        if r.importance >= min_imp and (tier_filter is None or r.tier.value == tier_filter):
            by_tier[r.tier.value].append(r)

    # Format as markdown
    lines = [
        f"## 🧠 {agent_name.title()}'s Enhanced Memory Context",
        "",
        f"> Query: *{query}*",
        "",
    ]

    tier_emoji = {
        "working": "⚡",
        "episodic": "📖",
        "semantic": "🧩",
        "procedural": "⚙️",
    }
    tier_desc = {
        "working": "Active session / current context",
        "episodic": "Past experiences / events",
        "semantic": "Facts, concepts, knowledge",
        "procedural": "Rules, patterns, how-to",
    }

    for tier_name in ["semantic", "procedural", "episodic", "working"]:
        items = by_tier.get(tier_name, [])
        if not items:
            continue
        emoji = tier_emoji.get(tier_name, "•")
        lines.append(f"### {emoji} {tier_name.title()} — {tier_desc.get(tier_name, '')}")
        for r in items[:k]:
            imp_bar = "▓" * int(r.importance * 10) + "░" * (10 - int(r.importance * 10))
            source = _src(r) or "memory"
            lines.append(f"- [{imp_bar}] {r.text[:300]}{'...' if len(r.text) > 300 else ''}")
            lines.append(f"  _source: {source} | tier: {r.tier.value}_")
        lines.append("")

    if not any(by_tier.values()):
        lines.append("_No relevant memories found above importance threshold._")

    lines.append("---")
    lines.append("*This context was auto-generated by DuckBot's enhanced brain (brain_inflate).*")

    markdown_block = "\n".join(lines)

    total = sum(len(v) for v in by_tier.values())
    return {
        "context": markdown_block,
        "total_memories": total,
        "tiers_covered": [t for t, v in by_tier.items() if v],
        "query": query,
        "agent_name": agent_name,
    }


async def handle_brain_sync(args: dict) -> dict:
    """Sync stored memories back to agent context files.

    Supports two targets:
    - OpenClaw:  ~/.openclaw/workspace/memory/{MEMORY,USER,SOUL}.md
                 No char limits — rich markdown format.
    - Hermes:    ~/.hermes/memories/{MEMORY,USER}.md (char-limited, §-delimited)
                 ~/.hermes/SOUL.md (no limit)

    This is what makes the brain "enhanced" vs. just "storage":
    it actively maintains the files agents read at startup.
    """
    target = args.get("target", "openclaw")
    memory_k = args.get("memory_k", 20)
    user_k = args.get("user_k", 15)
    dry_run = args.get("dry_run", False)

    from pathlib import Path
    import os

    def _src(r) -> str:
        """Extract source_path from a QueryResult. QueryResult stores
        source_path in metadata (not as a direct attribute), so a naive
        r.source_path would AttributeError on every call. The previous
        version had 4 sites that all crashed on this — centralizing here
        means one fix covers all of them."""
        return (getattr(r, "metadata", None) or {}).get("source_path", "") or ""

    def _tier(r) -> str:
        return r.tier.value if hasattr(r.tier, "value") else r.tier

    def _imp(r) -> float:
        """Extract importance from a QueryResult. Same shape as _src —
        importance lives in metadata, not as a direct attribute."""
        v = (getattr(r, "metadata", None) or {}).get("importance", 0)
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0
    from datetime import datetime

    mem = Memory()
    written: dict[str, str] = {}

    def _dry(content: str) -> str:
        return f"[DRY RUN — not written] {len(content)} chars"

    all_tiers = [Tier.WORKING, Tier.EPISODIC, Tier.SEMANTIC, Tier.PROCEDURAL]
    tier_summaries: dict[str, list] = {t.value: [] for t in all_tiers}
    # The previous version called recall("", ...) — an empty query produced
    # essentially-random results. The semantic fallback "important memory"
    # gives the embedder + BM25 actual signal to rank by.
    for tier in all_tiers:
        results, _ = await mem.recall(
            "important memory", k=memory_k // 4 + 1, tier=tier.value, min_importance=0.2,
        )
        tier_summaries[tier.value] = results

    user_results, _ = await mem.recall(
        "user preferences habits identity profile personality", k=user_k, min_importance=0.2,
    )
    soul_results, _ = await mem.recall(
        "identity principles values rules behavior patterns", k=10, tier="procedural", min_importance=0.2,
    )

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ---------------------------------------------------------------------------
    # OpenClaw target — rich markdown, no char limits
    # ---------------------------------------------------------------------------
    if target in ("openclaw", "both"):
        openclaw_memory = Path(os.environ.get(
            "OPENCLAW_WORKSPACE_MEMORY",
            str(Path.home() / ".openclaw" / "workspace" / "memory"),
        ))
        openclaw_memory.mkdir(parents=True, exist_ok=True)

        tier_emoji = {"working": "⚡", "episodic": "📖", "semantic": "🧩", "procedural": "⚙️"}

        # MEMORY.md
        oc_mem_lines = [
            f"# DuckBot Enhanced Memory — {now}",
            "",
            "_Auto-generated by duckbot-rag-memory's enhanced brain. Do not edit manually._",
            "",
        ]
        for tier_name, items in tier_summaries.items():
            emoji = tier_emoji.get(tier_name, "•")
            oc_mem_lines.append(f"## {emoji} {tier_name.title()}")
            for r in items[:memory_k // 4]:
                imp = _imp(r)
                oc_mem_lines.append(
                    f"- **[{imp:.0%}]** {r.text[:400]}{'...' if len(r.text) > 400 else ''}"
                )
                if _src(r):
                    oc_mem_lines.append(f"  _src: {_src(r)}_")
            if not items:
                oc_mem_lines.append("_No memories in this tier._")
            oc_mem_lines.append("")
        oc_mem_content = "\n".join(oc_mem_lines)
        oc_mem_key = "openclaw/MEMORY.md"
        if dry_run:
            written[oc_mem_key] = _dry(oc_mem_content)
        else:
            (openclaw_memory / "MEMORY.md").write_text(oc_mem_content, encoding="utf-8")
            written[oc_mem_key] = str(openclaw_memory / "MEMORY.md")

        # USER.md
        seen_texts: set[str] = set()
        oc_user_lines = [f"# User Profile — {now}", "", "_What the enhanced brain knows about the user._", ""]
        for r in user_results:
            if r.text not in seen_texts:
                seen_texts.add(r.text)
                oc_user_lines.append(f"- {r.text[:500]}")
                # r.source_path lives in metadata on QueryResult, not as a
                # direct attribute. The previous code crashed on every
                # brain_sync call with AttributeError; fall back to the
                # metadata dict.
                src = (r.metadata or {}).get("source_path", "")
                tier_val = r.tier.value if hasattr(r.tier, "value") else r.tier
                if src:
                    oc_user_lines.append(f"  _src: {src} ({tier_val})_")
        if len(oc_user_lines) <= 3:
            oc_user_lines.append("_No user facts stored yet._")
        oc_user_content = "\n".join(oc_user_lines)
        oc_user_key = "openclaw/USER.md"
        if dry_run:
            written[oc_user_key] = _dry(oc_user_content)
        else:
            (openclaw_memory / "USER.md").write_text(oc_user_content, encoding="utf-8")
            written[oc_user_key] = str(openclaw_memory / "USER.md")

        # SOUL.md (procedural — identity + principles)
        oc_soul_lines = [f"# Agent Soul — {now}", "", "_Identity and principles from the procedural memory tier._", ""]
        for r in soul_results:
            oc_soul_lines.append(f"- {r.text[:500]}")
        if len(oc_soul_lines) <= 3:
            oc_soul_lines.append("_No soul memories yet._")
        oc_soul_content = "\n".join(oc_soul_lines)
        oc_soul_key = "openclaw/SOUL.md"
        if dry_run:
            written[oc_soul_key] = _dry(oc_soul_content)
        else:
            (openclaw_memory / "SOUL.md").write_text(oc_soul_content, encoding="utf-8")
            written[oc_soul_key] = str(openclaw_memory / "SOUL.md")

    # ---------------------------------------------------------------------------
    # Hermes target — char-limited §-delimited format per Hermes spec
    # https://hermes-agent.nousresearch.com/docs/user-guide/features/memory
    # ---------------------------------------------------------------------------
    if target in ("hermes", "both"):
        hermes_home = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes")))
        hermes_memories = hermes_home / "memories"
        hermes_memories.mkdir(parents=True, exist_ok=True)

        # Hermes MEMORY.md: 2,200 char limit, §-delimited entries
        # Format: each entry is a line, prefixed with §
        def _make_hermes_entry(r: "RecallResult") -> str:
            imp = int((_imp(r)) * 10)
            bar = "▓" * imp + "░" * (10 - imp)
            src = f" [{_src(r) or 'memory'}]" if _src(r) else ""
            return f"§[{bar}]{r.text[:300]}{src}"

        hm_mem_entries: list[str] = []
        for tier_name, items in tier_summaries.items():
            for r in items:
                entry = _make_hermes_entry(r)
                if sum(len(e) for e in hm_mem_entries) + len(entry) + 60 > 2200:
                    # This entry would overflow the 2200-char budget. Skip it
                    # and try the next (which may be smaller and fit), rather
                    # than abandoning the entire tier — and, via the old
                    # for/else+break pattern, every tier after it.
                    continue
                hm_mem_entries.append(entry)

        hm_mem_lines = [f"MEMORY [{len(''.join(hm_mem_entries))}/2200 chars]"]
        hm_mem_lines.extend(hm_mem_entries or ["§No memories stored yet."])
        hm_mem_content = "\n".join(hm_mem_lines)
        hm_mem_key = "hermes/MEMORY.md"
        if dry_run:
            written[hm_mem_key] = _dry(hm_mem_content)
        else:
            (hermes_memories / "MEMORY.md").write_text(hm_mem_content, encoding="utf-8")
            written[hm_mem_key] = str(hermes_memories / "MEMORY.md")

        # Hermes USER.md: 1,375 char limit, §-delimited entries
        seen: set[str] = set()
        hm_user_entries: list[str] = []
        for r in user_results:
            if r.text in seen:
                continue
            seen.add(r.text)
            src = f" [{_src(r) or 'memory'}]" if _src(r) else ""
            entry = f"§{r.text[:250]}{src}"
            if sum(len(e) for e in hm_user_entries) + len(entry) + 60 > 1375:
                break
            hm_user_entries.append(entry)

        hm_user_lines = [f"USER PROFILE [{len(''.join(hm_user_entries))}/1375 chars]"]
        hm_user_lines.extend(hm_user_entries or ["§No user facts stored yet."])
        hm_user_content = "\n".join(hm_user_lines)
        hm_user_key = "hermes/USER.md"
        if dry_run:
            written[hm_user_key] = _dry(hm_user_content)
        else:
            (hermes_memories / "USER.md").write_text(hm_user_content, encoding="utf-8")
            written[hm_user_key] = str(hermes_memories / "USER.md")

        # Hermes SOUL.md: no hard char limit — at ~/.hermes/SOUL.md (global personality)
        # https://hermes-agent.nousresearch.com/docs/user-guide/features/personality
        hm_soul_lines = [
            f"# Agent Soul — synced from duckbot-rag-memory procedural tier — {now}",
            "",
            "_Do not edit manually — this file is auto-generated and will be overwritten on next sync._",
            "",
        ]
        for r in soul_results:
            hm_soul_lines.append(f"§ {r.text[:500]}")
        if len(hm_soul_lines) <= 3:
            hm_soul_lines.append("_No soul memories yet._")
        hm_soul_content = "\n".join(hm_soul_lines)
        hm_soul_key = "hermes/SOUL.md"
        if dry_run:
            written[hm_soul_key] = _dry(hm_soul_content)
        else:
            (hermes_home / "SOUL.md").write_text(hm_soul_content, encoding="utf-8")
            written[hm_soul_key] = str(hermes_home / "SOUL.md")

    return {
        "files": written,
        "dry_run": dry_run,
        "target": target,
        "total_memories_synced": sum(len(v) for v in tier_summaries.values()),
    }


async def handle_brain_wake_up(args: dict) -> dict:
    """One-call session-startup context load. MemPalace-inspired.

    Returns a single dict with:
      - memories: top-k recent recall results (dropped superseded chunks)
      - blocks: active memory blocks (preview only — char-bounded)
      - graph_summary: top-10 entities + total count
      - fsrs_review_queue: chunks due for review (max 5)
      - stats: brief store counts per tier

    Drop-in for agents that want a single MCP call on session start
    instead of N round-trips. Query-less mode = "what was I doing recently?"
    """
    brain = Brain()
    return brain.wake_up(
        query=args.get("query"),
        k=args.get("k", 8),
        include_blocks=args.get("include_blocks", True),
        include_graph=args.get("include_graph", True),
        include_fsrs_review=args.get("include_fsrs_review", True),
    )


async def handle_brain_index(args: dict) -> dict:
    """Compact whole-corpus summary using the AAAK dialect (dialect.py).

    The LLM scans the output in <500 tokens and picks entries to expand
    via brain_recall. Default 5000 lines; pair with --tier to narrow.
    Returns the compressed string + a small parse hint (total count).
    """
    from src.dialect import compress_corpus
    from src.tier import Tier
    tier = args.get("tier")
    if tier is not None:
        tier = Tier(tier)
    max_chunks = int(args.get("max_chunks", 5000))
    preview_chars = int(args.get("preview_chars", 80))

    # Walk every tier's collection. We could use the recall path, but
    # the whole point of the index is to be cheaper than recall — pull
    # documents+metadatas directly with a single .get() per tier.
    from src.memory import Memory
    mem = Memory()
    # We need the store, but _ensure_initialized is async. Run it.
    import asyncio
    store, _ = await mem._ensure_initialized()
    chunks: list[dict] = []
    tier_filter = tier
    for t in store.all_collections:
        coll = store.collection_for(t)
        if tier_filter is not None and t != tier_filter:
            continue
        try:
            data = coll.get(limit=max_chunks, include=["documents", "metadatas"])
        except Exception:
            continue
        ids = (data or {}).get("ids") or []
        docs = (data or {}).get("documents") or []
        metas = (data or {}).get("metadatas") or []
        for cid, doc, md in zip(ids, docs, metas):
            md = md or {}
            # Skip superseded chunks — wake_up's behavior, applied here too.
            if md.get("superseded_by"):
                continue
            chunks.append({
                "text": doc or "",
                "tier": t,
                "importance": md.get("importance", 0.0),
                "source_path": md.get("source_path", ""),
            })
            if len(chunks) >= max_chunks:
                break
        if len(chunks) >= max_chunks:
            break

    compressed = compress_corpus(chunks, max_chunks=max_chunks, preview_chars=preview_chars)
    return {
        "dialect": compressed,
        "total": len(chunks),
        "preview_chars": preview_chars,
        "tier_filter": tier.value if tier else None,
    }


async def handle_brain_nudge(args: dict) -> dict:
    """Proactive memory nudge: surface stale-but-important memories.

    Algorithm:
      1. Recall high-importance memories (above --min_importance).
      2. Filter to ones whose last_recalled_at is older than --stale_days
         (or never recalled).
      3. If --context is provided, bias toward memories that match the
         current focus via a second recall.
      4. Drop superseded chunks.
      5. Return top --k with a "reason" explaining why each is being
         surfaced.

    Use cases:
      - Cron: schedule daily to remind the agent of forgotten decisions.
      - Mid-task: call before starting a new chunk of work to surface
        related prior context the agent may not have on its mind.
    """
    from src.memory import Memory
    import time
    mem = Memory()
    k = int(args.get("k", 5))
    min_imp = float(args.get("min_importance", 0.6))
    stale_days = int(args.get("stale_days", 7))
    context = args.get("context")
    stale_cutoff = time.time() - (stale_days * 86400.0)

    # 1 + 2: pull candidates and filter
    store, _ = await mem._ensure_initialized()
    candidates: list[dict] = []
    for t in store.all_collections:
        try:
            data = store.collection_for(t).get(
                include=["documents", "metadatas"], limit=200,
            )
        except Exception:
            continue
        ids = (data or {}).get("ids") or []
        docs = (data or {}).get("documents") or []
        metas = (data or {}).get("metadatas") or []
        for cid, doc, md in zip(ids, docs, metas):
            md = md or {}
            if md.get("superseded_by"):
                continue
            try:
                imp = float(md.get("importance", 0.0))
            except (TypeError, ValueError):
                imp = 0.0
            if imp < min_imp:
                continue
            last_recall = float(md.get("last_recalled_at") or 0.0)
            # Either never recalled OR last recall is older than cutoff
            if last_recall >= stale_cutoff:
                continue
            candidates.append({
                "chunk_id": cid,
                "text": doc or "",
                "tier": t,
                "importance": imp,
                "source_path": md.get("source_path", ""),
                "last_recalled_at": last_recall,
                "age_days": round((time.time() - last_recall) / 86400.0, 1) if last_recall else None,
            })

    # 3: context bias — if provided, re-rank by recall similarity
    if context and candidates:
        try:
            results, _ = await mem.recall(context, k=k * 3, tier=None, min_importance=min_imp)
            # Build a relevance map: chunk_id -> rrf_score
            rel: dict[str, float] = {}
            for r in results:
                md = getattr(r, "metadata", None) or {}
                # The hybrid_query returns QueryResult objects whose
                # metadata is the chunk's metadata dict.
                cid = r.chunk_id
                rel[cid] = float(getattr(r, "rrf_score", 0.0) or 0.0)
            for c in candidates:
                c["relevance"] = rel.get(c["chunk_id"], 0.0)
            candidates.sort(key=lambda c: (c["relevance"], c["importance"]), reverse=True)
        except Exception:
            # Fall back to importance-only ordering
            candidates.sort(key=lambda c: c["importance"], reverse=True)
    else:
        candidates.sort(key=lambda c: c["importance"], reverse=True)

    out = candidates[:k]
    # 4: build reasons
    for c in out:
        reasons = []
        if c["importance"] >= 0.8:
            reasons.append(f"high importance ({c['importance']:.2f})")
        if c["age_days"] is None:
            reasons.append("never recalled")
        elif c["age_days"] >= stale_days * 2:
            reasons.append(f"last recalled {c['age_days']}d ago")
        else:
            reasons.append(f"last recalled {c['age_days']}d ago")
        if "relevance" in c and c["relevance"] > 0:
            reasons.append("matches current context")
        c["reason"] = " · ".join(reasons) if reasons else "stale"
        # Truncate text preview
        c["preview"] = (c.pop("text") or "")[:200]

    return {
        "nudges": out,
        "total_candidates": len(candidates),
        "context": context,
        "stale_days": stale_days,
        "min_importance": min_imp,
    }


async def handle_brain_skill_create(args: dict) -> dict:
    """Auto-generate an agentskills.io-compatible SKILL.md from a task
    description. Writes to skills/<slug>/SKILL.md.

    Why this exists: when an agent solves a new task successfully, the
    win should be reusable. The skill is the unit of re-use. This tool
    lets the agent (or an external LLM script) crystallize the win
    into a discoverable manifest.

    Pure templating — no LLM call by default. An LLM-driven path could
    route through LM Studio for higher-quality bodies; for v0.1 the
    deterministic template is enough and the user can edit the file
    afterwards.
    """
    from src.skillgen import write_skill, render_from_memory

    name = args.get("name")
    description = args.get("description")
    if not name or not description:
        return {"error": "name and description are required"}
    instructions = args.get("instructions") or []
    if not instructions:
        return {"error": "instructions list is required (at least one step)"}

    body = render_from_memory(
        name=name,
        description=description,
        instructions=instructions,
        example=args.get("example", ""),
        emoji=args.get("emoji"),
    )

    # Skills dir is the repo's skills/ directory. The MCP server runs
    # from the repo root so we can resolve it relative to the file.
    repo_root = Path(__file__).resolve().parent.parent
    skills_dir = repo_root / "skills"
    try:
        path = write_skill(
            skills_dir=skills_dir,
            name=name,
            description=description,
            body_markdown=body,
            emoji=args.get("emoji"),
            overwrite=bool(args.get("overwrite", False)),
        )
        return {
            "path": str(path),
            "slug": path.parent.name,
            "created": True,
        }
    except FileExistsError as e:
        return {"error": str(e), "created": False, "hint": "pass overwrite=true to replace"}


async def handle_brain_user_model(args: dict) -> dict:
    """Aggregate user-related facts into a single memory block.

    Honcho-inspired user modeling: instead of forcing the agent to
    re-read every fact about the user on session start, we periodically
    distill the high-importance facts into one consolidated block.

    The block content is built from:
      1. Memories that mention the user entity (Duckets / user / he / she)
      2. High-importance memories with user-related keywords
      3. Entity-relationship triples where the user is on either end

    The model is updated in-place via block_write. Older content is
    preserved (we APPEND a new section, not overwrite) so the model
    accumulates over time. The agent can prune by editing the block
    directly.
    """
    from src.memory import Memory
    from src.tier import Tier
    from src.blocks import BlockStore
    from src.connectors.base import Brain

    block_name = args.get("block_name", "user")
    min_imp = float(args.get("min_importance", 0.5))
    max_facts = int(args.get("max_facts", 30))
    k_per_query = int(args.get("k_per_query", 50))

    mem = Memory()
    store, _ = await mem._ensure_initialized()

    # 1. Pull user-related candidates across all tiers.
    seen_texts: set[str] = set()
    candidates: list[dict] = []
    user_queries = [
        "Duckets preferences",
        "Duckets routine",
        "Duckets habits",
        "user profile",
    ]
    for q in user_queries:
        try:
            results, _ = await mem.recall(
                q, k=k_per_query // len(user_queries) + 1, tier=None,
                min_importance=min_imp,
            )
        except Exception:
            continue
        for r in results:
            md = getattr(r, "metadata", None) or {}
            text = (getattr(r, "text", "") or "").strip()
            if not text or text in seen_texts:
                continue
            if md.get("superseded_by"):
                continue
            seen_texts.add(text)
            candidates.append({
                "text": text,
                "tier": md.get("tier", "unknown"),
                "importance": md.get("importance", 0.0),
                "source_path": md.get("source_path", ""),
            })
            if len(candidates) >= max_facts:
                break
        if len(candidates) >= max_facts:
            break

    # 2. Sort by importance desc, take top max_facts
    candidates.sort(key=lambda c: c["importance"], reverse=True)
    candidates = candidates[:max_facts]

    # 3. Build the block content
    from datetime import date
    today = date.today().isoformat()
    lines = [
        f"# User Model (auto-generated {today})",
        "",
        f"_Consolidated from {len(candidates)} high-importance user-related facts._",
        "",
        "## Facts",
        "",
    ]
    for c in candidates:
        tier = c["tier"]
        imp = c["importance"]
        src = c["source_path"] or ""
        text = c["text"][:300].replace("\n", " ")
        lines.append(f"- **[{tier}]** (imp {imp:.2f}) {text}")
        if src:
            lines.append(f"  _src: {src}_")
    lines.append("")

    new_section = "\n".join(lines)

    # 4. Append to the existing block (don't destroy history).
    brain = Brain()
    existing = brain.block_read(block_name)
    if existing and existing.get("text"):
        combined = existing["text"].rstrip() + "\n\n" + new_section
    else:
        combined = new_section

    result = brain.block_write(block_name, combined)
    return {
        "block_name": block_name,
        "facts_aggregated": len(candidates),
        "char_count": len(combined),
        "result": result,
    }


async def handle_brain_palace(args: dict) -> dict:
    """Wing/Room/Drawer 2D view of the brain.

    With no args: list every wing (person/project) with room counts.
    With --wing: return the drawers in that wing, optionally filtered
    to one --room (date) and/or --tier. Use this for project-scoped
    recall: 'everything I know about OpenClaw from this week' without
    manually filtering by source_path.
    """
    from src.palace import PalaceIndex
    from src.memory import Memory
    mem = Memory()
    store, _ = await mem._ensure_initialized()
    pi = PalaceIndex.from_store(store)

    wing = args.get("wing")
    if wing:
        room = args.get("room")
        tier = args.get("tier")
        max_drawers = int(args.get("max_drawers", 100))
        drawers = pi.walk(wing, room=room, tier=tier, max_drawers=max_drawers)
        return {
            "wing": wing,
            "room": room,
            "tier": tier,
            "drawer_count": len(drawers),
            "drawers": [d.to_dict() for d in drawers],
        }

    # No wing: list everything
    wings = pi.wings()
    return {
        "wing_count": len(wings),
        "total_drawers": len(pi.all_drawers()),
        "wings": [w.to_dict() for w in wings],
    }


HANDLERS = {
    "remember": handle_remember,
    "recall": handle_recall,
    "reflect": handle_reflect,
    "forget": handle_forget,
    "stats": handle_stats,
    "watch": handle_watch,
    "doctor": handle_doctor,
    # v0.10.0 — useful MCP tools extension
    "recall_verbatim": handle_recall_verbatim,
    "fsrs_review": handle_fsrs_review,
    "decay_status": handle_decay_status,
    "forget_by_query": handle_forget_by_query,
    "search_verbatim": handle_search_verbatim,
    # v0.11.0 — OpenClaw dreaming bridge + Hermes /learn + Active Memory
    "dreaming_read": handle_dreaming_read,
    "dreaming_cycle": handle_dreaming_cycle,
    "learn": handle_learn,
    "active_memory": handle_active_memory,
    # v0.11.2 — Enhanced Brain: context inflation
    "brain_inflate": handle_brain_inflate,
    "brain_wake_up": handle_brain_wake_up,
    "brain_index": handle_brain_index,
    "brain_nudge": handle_brain_nudge,
    "brain_skill_create": handle_brain_skill_create,
    "brain_user_model": handle_brain_user_model,
    "brain_palace": handle_brain_palace,
    "brain_sync": handle_brain_sync,
}

# Register the 28 connector tools (graph + blocks + quarantine + scan +
# brain_* aliases). Called after HANDLERS is defined so the dispatch table
# is ready.
_register_connector_tools()


# -----------------------------------------------------------------------------
# MCP stdio transport
# -----------------------------------------------------------------------------

def mcp_stdio():
    """Run the MCP server on stdio. JSON-RPC 2.0 protocol.

    Stdout/stderr are forced to line-buffered, binary-untouched mode on
    startup. Without this, Windows block-buffers the subprocess stdout
    (4-8 KiB chunks) and short ``initialize`` responses (<200 bytes) get
    stuck in the kernel pipe buffer, causing MCP clients to time out
    with "Connection closed" before the first response arrives.
    ``reconfigure(line_buffering=True)`` (py3.7+) is cross-platform
    and is a no-op on platforms that already flush per-write, so this
    is safe on macOS and Linux too. See:
    https://docs.python.org/3/library/sys.html#sys.stdout.reconfigure
    """
    import sys as _s
    try:
        # ``line_buffering=True`` flushes after every newline.
        # ``write_through=True`` bypasses any parent TextIOWrapper.
        for stream in (_s.stdin, _s.stdout, _s.stderr):
            # reconfigure() is unavailable on some embedded Pythons / odd
            # stdio wrappers; guard it so a benign host quirk never
            # blocks the server from starting.
            try:
                stream.reconfigure(line_buffering=True)
            except (AttributeError, ValueError, OSError):
                pass
    except Exception:
        # Never let a stdio-reconfigure edge case abort startup; the
        # server should still answer (just possibly slower).
        pass
    # v0.11.7: keep one event loop alive for the whole server lifetime.
    # The previous version called `asyncio.run(handler(args))` per tool
    # call, which creates + tears down an event loop on every request.
    # That's expensive (shared httpx clients get orphaned and re-created,
    # any loop-bound resources break) and unnecessary — mcp_stdio() is
    # itself synchronous and we can drive handlers from a single loop.
    _server_loop = asyncio.new_event_loop()
    try:
        while True:
            try:
                line = _s.stdin.readline()
                if not line:
                    break
                msg = json.loads(line)
                method = msg.get("method")
                mid = msg.get("id")
                if method == "initialize":
                    resp = {
                        "jsonrpc": "2.0", "id": mid,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "serverInfo": {"name": "duckbot-memory", "version": "0.11.7"},
                            "capabilities": {"tools": {}},
                        },
                    }
                elif method == "notifications/initialized":
                    continue
                elif method == "tools/list":
                    resp = {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
                elif method == "tools/call":
                    params = msg.get("params", {})
                    name = params.get("name")
                    args = params.get("arguments", {})
                    handler = HANDLERS.get(name)
                    if not handler:
                        resp = {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"unknown tool: {name}"}}
                    else:
                        try:
                            result = _server_loop.run_until_complete(handler(args))
                            resp = {
                                "jsonrpc": "2.0", "id": mid,
                                "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]},
                            }
                        except Exception as exc:
                            resp = {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": str(exc)}}
                else:
                    resp = {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"unknown method: {method}"}}
                _s.stdout.write(json.dumps(resp) + "\n")
                _s.stdout.flush()
            except json.JSONDecodeError:
                continue
            except KeyboardInterrupt:
                break
            except Exception as exc:
                err = {"jsonrpc": "2.0", "error": {"code": -32603, "message": str(exc)}}
                try:
                    _s.stdout.write(json.dumps(err) + "\n")
                    _s.stdout.flush()
                except Exception:
                    pass
    finally:
        # Close the long-lived event loop on server exit. Cancels any
        # pending tasks and releases the loop-bound shared httpx client.
        try:
            _server_loop.close()
        except Exception:
            pass


def main():
    p = argparse.ArgumentParser(description="DuckBot memory MCP server")
    p.add_argument(
        "--http", type=int, default=None,
        help="HTTP mode is not yet implemented — accepted for forward "
             "compatibility but exits with a clear error if used.",
    )
    args = p.parse_args()
    if args.http is not None:
        print(
            f"Error: HTTP mode (--http {args.http}) is not yet implemented. "
            "Use stdio: `python -m src.mcp_server` (no flag), then point "
            "your MCP client at `scripts/duckbot-memory-mcp.sh` or `.bat`. "
            "Tracked in CHANGELOG.",
            file=sys.stderr,
        )
        sys.exit(2)  # 2 = misuse, not 1 = generic error
    mcp_stdio()


if __name__ == "__main__":
    main()
