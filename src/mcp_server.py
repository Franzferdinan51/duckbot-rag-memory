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
            source = r.source_path or "memory"
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
                imp = r.importance or 0
                oc_mem_lines.append(
                    f"- **[{imp:.0%}]** {r.text[:400]}{'...' if len(r.text) > 400 else ''}"
                )
                if r.source_path:
                    oc_mem_lines.append(f"  _src: {r.source_path}_")
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
                if r.source_path:
                    oc_user_lines.append(f"  _src: {r.source_path} ({r.tier.value})_")
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
            imp = int((r.importance or 0) * 10)
            bar = "▓" * imp + "░" * (10 - imp)
            src = f" [{r.source_path or 'memory'}]" if r.source_path else ""
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
            src = f" [{r.source_path or 'memory'}]" if r.source_path else ""
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
