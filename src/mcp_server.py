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
import time
from pathlib import Path

# Add parent to path so this can be run as `python -m src.mcp_server`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import __version__ as PACKAGE_VERSION
from src.memory import Memory
from src.tier import Tier


# -----------------------------------------------------------------------------
# Tool definitions (MCP format)
# -----------------------------------------------------------------------------

TOOLS = [
    {
        "name": "remember",
        "description": "Save a memory. Auto-chunks, classifies tier, extracts entities, embeds, stores. Returns chunk_id, tier, importance. Pass kind='skill_candidate' to stamp a lightweight skill candidate (agent-driven pipeline, no LLM, stored in procedural tier with metadata.kind='skill_candidate').",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "the memory content"},
                "source_path": {"type": "string", "description": "where this came from (e.g. file path, conversation id)", "default": "<remember>"},
                "metadata": {"type": "object", "description": "arbitrary metadata to attach"},
                "force_tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"], "description": "override auto tier"},
                "kind": {"type": "string", "enum": ["skill_candidate"], "description": "special mode: stamp a skill candidate chunk (no LLM, stored in procedural tier for later promotion via brain_skills_promote)"},
                "summary": {"type": "string", "description": "short label for skill candidates"},
                "importance": {"type": "number", "default": 0.6, "description": "0..1 ranking score for skill candidates"},
                "trust_level": {"type": "string", "enum": ["full", "standard"], "default": "full", "description": "trust_level='full' (default) skips the injection scan. 'standard' runs the scan and quarantines suspicious content."},
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
                "tier_priors": {"type": "boolean", "default": False, "description": "Layer 11: apply per-tier multiplicative weights"},
                "tier_priors_overrides": {"type": "object", "description": "per-tier weight overrides, e.g. {\"procedural\": 2.0}"},
                "fsrs": {"type": "boolean", "default": False, "description": "Layer 9: use FSRS-6 power-law forgetting instead of Ebbinghaus"},
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
        "name": "brain_decay_apply",
        "description": "Prune chunks whose Ebbinghaus retention R has dropped below `retention_floor`. The 'memory decay' cron job. Public-domain math (1885); no LLM call. Default `dry_run=True` so the caller can preview; pass `dry_run=False` to actually delete. Use on a daily schedule to keep the episodic tier from growing unbounded.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"], "description": "limit to one tier (default: all)"},
                "retention_floor": {"type": "number", "default": 0.05, "description": "chunks with R < floor are pruned"},
                "max_prune": {"type": "integer", "default": 1000, "description": "safety cap on chunks deleted in one call"},
                "dry_run": {"type": "boolean", "default": True, "description": "preview only — don't actually delete"},
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
        "name": "brain_skills_list",
        "description": "List unpromoted skill-candidate chunks (agent-driven pipeline). Returns candidates stamped via remember(kind='skill_candidate'), sorted by recency then importance. The AGENT reads this list and decides which to promote — the brain does no LLM work. Pass include_promoted=true to also see already-promoted candidates.",
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
        "description": "Semantic top-N skill candidates matching a query (agent-driven pipeline). Uses hybrid retrieval scoped to the procedural tier, then filters to unpromoted skill candidates. Use this when the agent is working on a topic and wants to know 'are there candidate skills about X?' — the agent reads the results and decides which to promote. No LLM call; just vector + BM25.",
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
        "description": "Promote a skill candidate to a full agentskills.io SKILL.md. The AGENT authors name/description/instructions using its own LLM context — the brain is pure storage + template (no LLM). Writes skills/<slug>/SKILL.md and marks the candidate chunk as promoted. Pass overwrite=true to re-promote or replace an existing skill file.",
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
        "description": "Wing/Room/Drawer 2D view of the brain (MemPalace-inspired). With no args, lists every wing (person/project) and its room count, cross-referenced to the 'user' memory block so the agent sees which wings are already covered by the user model (modeled_in_user_block: true/false per wing). With --wing, returns the drawers in that wing, optionally filtered to one room and/or tier. Use this for project-scoped recall ('everything I know about OpenClaw from this week') without manually filtering by source_path.",
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
    {
        "name": "brain_optimize_fsrs",
        "description": "Self-tune the FSRS-6 forgetting-curve exponent (w20) from the brain's recall history. Grid-searches w20 over the live chunk set, minimizing MSE between predicted R(t, S) and observed 'remembered' (recall_count>0) vs 'forgotten' (never recalled) labels. Returns the proposed w20 + improvement vs the current default. Does NOT auto-apply — call brain_apply_fsrs_w20 to commit. Run on a weekly cron.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "default_w20": {"type": "number", "default": 0.9, "description": "comparison baseline (typically the shipped 0.9)"},
                "w20_lo": {"type": "number", "default": 0.05, "description": "search grid low"},
                "w20_hi": {"type": "number", "default": 3.0, "description": "search grid high"},
                "w20_step": {"type": "number", "default": 0.05, "description": "search grid step"},
            },
        },
    },
    {
        "name": "brain_apply_fsrs_w20",
        "description": "Apply a chosen w20 to the brain's fsrs.DEFAULT_W20. Call this after brain_optimize_fsrs returns a better w20. The change is in-process only — restart the server to persist, or set DUCKBOT_FSRS_W20=... env var for persistence.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "w20": {"type": "number", "description": "the new w20 to use"},
            },
            "required": ["w20"],
        },
    },
    {
        "name": "brain_fsrs_optimize_apply",
        "description": "One-call: fit w20 from recall history, then apply it if it improves the MSE by at least `min_improvement_pct` over the current default. Returns the optimization result + whether it was applied. Use on a weekly cron — keeps the forgetting curve tuned to the brain's actual usage pattern.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "min_improvement_pct": {"type": "number", "default": 1.0, "description": "minimum improvement over baseline (percent) to apply"},
                "w20_lo": {"type": "number", "default": 0.05},
                "w20_hi": {"type": "number", "default": 3.0},
                "w20_step": {"type": "number", "default": 0.05},
            },
        },
    },
    {
        "name": "brain_export",
        "description": "Export the entire brain as a single markdown file. One section per chunk, grouped by tier, with provenance. Use this to back up the brain, migrate to another machine, share with another agent, or inspect the corpus in a text editor. Default path is data/brain_export.md. Idempotent: re-running overwrites the same file. The reverse operation is brain_import — together they let you round-trip the brain as plain markdown.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "out_path": {"type": "string", "description": "where to write the export (default data/brain_export.md)"},
                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"], "description": "export only one tier (default: all)"},
                "include_superseded": {"type": "boolean", "default": False, "description": "include chunks marked superseded_by (default: skip — keeps the export clean)"},
            },
        },
    },
    {
        "name": "brain_import",
        "description": "Import a markdown file into the brain. Each top-level section (## Heading) becomes one remembered chunk; the section text is the content; the section heading is auto-detected for tier (semantic if 'rule/preference/decision', episodic if 'YYYY-MM-DD', procedural if 'how to/always/never', else working). Use this to ingest chat-history exports, project docs, or another brain's brain_export dump. Source paths are stamped so brain_recall can cite provenance.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "in_path": {"type": "string", "description": "path to the markdown file to import"},
                "source_path": {"type": "string", "description": "stamped as source_path on every chunk (default: filename)"},
            },
            "required": ["in_path"],
        },
    },
    {
        "name": "brain_seed_demo",
        "description": "Seed the brain with a small bundled sample corpus so a new user can see a working brain without ingesting their own files first. Idempotent: skips chunks that already exist. Useful for demos, smoke tests, and onboarding.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "force": {"type": "boolean", "default": False, "description": "re-seed even if chunks already exist"},
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
        # Add the tool schemas ONLY for tools that aren't already in the
        # canonical TOOLS list (e.g. brain_block_*, brain_graph_*,
        # brain_seed_blocks). For tools that already have a native
        # MCP handler (brain_remember, brain_recall, brain_recall_verbatim,
        # etc.), the canonical handler wins so the v0.14.0+
        # validation + skill-candidate support is honored.
        for t in TOOL_DEFINITIONS:
            if t["name"] not in {existing["name"] for existing in TOOLS}:
                TOOLS.append(t)
            # Always register the connector handler if we don't already
            # have a native one — otherwise the legacy (often-invalidating)
            # handler shadows the new validation-aware handler.
            if t["name"] not in HANDLERS:
                # The MCP main loop calls `run_until_complete(handler(args))`,
                # so the handler must return a coroutine / awaitable.
                # The legacy connector's `handle()` is SYNC — wrap it so
                # the returned lambda IS awaitable. Without this wrap,
                # every connector tool raised
                # "object dict can't be used in 'await' expression" when
                # called via tools/call.
                HANDLERS[t["name"]] = (
                    lambda args, h=_handle, n=t["name"]: _wrap_sync(h, n, args)
                )
    except Exception as _e:
        import sys as _s
        print(f"[mcp_server] connector tools not loaded: {_e}", file=_s.stderr)


async def _wrap_sync(handler, name: str, args: dict) -> dict:
    """Adapter: wrap a sync connector handler in an async coroutine so
    the MCP main loop's `run_until_complete()` can await it."""
    return handler(name, args)


# -----------------------------------------------------------------------------
# Tool implementations
# -----------------------------------------------------------------------------

async def handle_remember(args: dict) -> dict:
    from src.tier import Tier

    # Reject empty / whitespace-only text — same fix as the shared dispatch
    # in src/extensions/tools.py. Without this, an empty remember() either
    # silently fails (Chroma rejects empty documents) or stores a useless
    # chunk with chunk_id = sha256("" + source) forever.
    text = args.get("text") or ""
    if not text.strip():
        return {"error": "text must be a non-empty string"}

    # Agent-driven skill pipeline: stamp a candidate (no LLM).
    if args.get("kind") == "skill_candidate":
        from src.skill_pipeline import stamp_skill_candidate
        from src.connectors.base import Brain
        trust_level = args.get("trust_level", "full")
        if trust_level not in ("full", "standard"):
            return {"error": f"trust_level must be 'full' or 'standard', got {trust_level!r}"}
        result = stamp_skill_candidate(
            text=text,
            source=args.get("source_path", "agent://skill-candidate"),
            summary=args.get("summary", ""),
            importance=float(args.get("importance", 0.6)),
            trust_level=trust_level,
            brain=Brain(),
        )
        return {
            "chunk_id": result.chunk_id,
            "tier": result.tier,
            "kind": "skill_candidate",
            "stored": result.stored,
        }

    mem = Memory()
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
    if not args.get("query"):
        return {"error": "query is required"}
    mem = Memory()
    tpo = args.get("tier_priors_overrides")
    if tpo is not None and not isinstance(tpo, dict):
        return {"error": "tier_priors_overrides must be a dict"}
    results, stats = await mem.recall(
        args["query"],
        k=args.get("k", 5),
        tier=args.get("tier"),
        min_importance=args.get("min_importance"),
        rerank=args.get("rerank"),
        decay=args.get("decay"),
        tier_priors=args.get("tier_priors"),
        tier_priors_overrides=tpo,
        fsrs=args.get("fsrs"),
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
    if not args.get("chunk_id"):
        return {"error": "chunk_id is required"}
    mem = Memory()
    from src.tier import Tier
    tier = Tier(args["tier"]) if args.get("tier") else None
    ok = await mem.forget(args["chunk_id"], tier=tier)
    return {"deleted": ok}


# ---- v0.10.0 — useful MCP tools extension ----

async def handle_recall_verbatim(args: dict) -> dict:
    if not args.get("query"):
        return {"error": "query is required"}
    from src.connectors.base import Brain
    brain = Brain()
    tpo = args.get("tier_priors_overrides")
    if tpo is not None and not isinstance(tpo, dict):
        return {"error": "tier_priors_overrides must be a dict"}
    return {"results": brain.recall_verbatim(
        query=args["query"],
        k=args.get("k", 5),
        tier=args.get("tier"),
        min_importance=args.get("min_importance"),
        rerank=args.get("rerank"),
        decay=args.get("decay"),
        tier_priors=args.get("tier_priors"),
        tier_priors_overrides=tpo,
        fsrs=args.get("fsrs"),
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


async def handle_brain_decay_apply(args: dict) -> dict:
    """Prune chunks whose Ebbinghaus retention R has dropped below
    `retention_floor`. The 'memory decay' cron job — run on a daily
    schedule to keep the episodic tier from growing unbounded.
    Public-domain math (1885); no LLM call.
    """
    from src.connectors.base import Brain
    brain = Brain()
    return brain.decay_apply(
        tier=args.get("tier"),
        retention_floor=float(args.get("retention_floor", 0.05)),
        max_prune=int(args.get("max_prune", 1000)),
        dry_run=bool(args.get("dry_run", True)),
    )


async def handle_forget_by_query(args: dict) -> dict:
    if not args.get("query"):
        return {"error": "query is required"}
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
    # Reject empty / whitespace-only needle — same fix as the shared
    # surface's brain_search_verbatim. Otherwise every chunk trivially
    # matches "" via Python's `in` check.
    needle = (args.get("needle") or "").strip()
    if not needle:
        return {"error": "needle must be a non-empty string"}
    return {"matches": brain.search_verbatim(
        needle=needle, k=args.get("k", 5),
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
    if not args.get("text") or not str(args["text"]).strip():
        return {"error": "text must be a non-empty string"}
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
    if not args.get("tool"):
        return {"error": "tool is required"}
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
    if not args.get("query"):
        return {"error": "query is required"}
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
    from src.connectors.base import Brain
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


async def handle_brain_skills_list(args: dict) -> dict:
    """List skill-candidate chunks (agent-driven pipeline).

    Pure metadata scan over the procedural tier — no LLM call. Returns
    candidates sorted by recency then importance. The AGENT reads this
    list and decides which to promote via brain_skills_promote.
    """
    from src.skill_pipeline import list_candidates
    k = int(args.get("k", 50))
    if k <= 0:
        return {"error": "k must be a positive integer"}
    result = list_candidates(
        include_promoted=bool(args.get("include_promoted", False)),
        k=k,
    )
    # If list_candidates returned an error dict (e.g. k <= 0 from the
    # inner check), surface it as a top-level error rather than wrapping
    # in {"candidates": {"error": ...}} which is confusing.
    if isinstance(result, dict) and result.get("error"):
        return result
    return {"candidates": result}


async def handle_brain_skills_suggest(args: dict) -> dict:
    """Semantic top-N skill candidates matching a query (agent-driven pipeline).

    Uses hybrid retrieval (vector + BM25 + RRF) scoped to the procedural
    tier, then filters to unpromoted skill candidates. Returns top-k
    candidates sorted by semantic similarity score. No LLM call.
    """
    if not args.get("query") or not str(args["query"]).strip():
        return {"error": "query must be a non-empty string"}
    from src.skill_pipeline import suggest_candidates
    return {"candidates": suggest_candidates(
        query=args["query"],
        k=int(args.get("k", 5)),
    )}


async def handle_brain_skills_promote(args: dict) -> dict:
    """Promote a skill candidate to a full SKILL.md (agent-driven).

    The AGENT authors name/description/instructions using its own LLM
    context — the brain is pure storage + template (no LLM). Writes
    skills/<slug>/SKILL.md and marks the candidate chunk as promoted.
    """
    # Validate required fields upfront. Either `instructions` (flat list)
    # OR `instructions_markdown` (rich markdown body) must be provided —
    # the former for simple skills, the latter for richer bodies.
    if not args.get("chunk_id") or not args.get("name") or not args.get("description"):
        return {"error": "chunk_id, name, description are required"}
    if not (args.get("instructions") or args.get("instructions_markdown")):
        return {"error": "either instructions (list) or instructions_markdown (string) is required"}
    from src.skill_pipeline import promote_candidate
    return promote_candidate(
        chunk_id=args["chunk_id"],
        name=args["name"],
        description=args["description"],
        instructions=args.get("instructions") or [],
        example=args.get("example", ""),
        emoji=args.get("emoji"),
        overwrite=bool(args.get("overwrite", False)),
        instructions_markdown=args.get("instructions_markdown"),
    )


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

    With no args: list every wing (person/project) with room counts,
    cross-referenced to the 'user' block so the agent sees which
    wings the user model has already covered.
    With --wing: return the drawers in that wing, optionally filtered
    to one --room (date) and/or --tier. Use this for project-scoped
    recall: 'everything I know about OpenClaw from this week' without
    manually filtering by source_path.
    """
    from src.palace import PalaceIndex
    from src.memory import Memory
    from src.connectors.base import Brain
    mem = Memory()
    store, _ = await mem._ensure_initialized()
    pi = PalaceIndex.from_store(store)

    # Cross-reference: which wings the user model has already covered?
    # Read the 'user' block and extract any wing names it mentions
    # (case-insensitive). Cheap heuristic but useful: the agent sees
    # at a glance which projects have been actively modeled.
    modeled_wings: set[str] = set()
    try:
        brain = Brain()
        user_block = brain.block_read("user")
        if user_block and user_block.get("text"):
            # Find any wing name mentioned in the user block.
            wing_names = {w.name for w in pi.wings()}
            text_lower = user_block["text"].lower()
            for wn in wing_names:
                if wn.lower() in text_lower:
                    modeled_wings.add(wn)
    except Exception:
        # Non-fatal — cross-reference is best-effort.
        pass

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
            "modeled_in_user_block": wing in modeled_wings,
        }

    # No wing: list everything
    wings = pi.wings()
    out_wings = []
    for w in wings:
        d = w.to_dict()
        d["modeled_in_user_block"] = w.name in modeled_wings
        out_wings.append(d)
    return {
        "wing_count": len(wings),
        "total_drawers": len(pi.all_drawers()),
        "wings": out_wings,
    }


async def handle_brain_optimize_fsrs(args: dict) -> dict:
    """Self-tune the FSRS-6 w20 (forgetting exponent) from recall history.

    Grid-searches w20 ∈ [w20_lo, w20_hi] at w20_step granularity,
    minimizing MSE between predicted R(t, S) and observed
    'remembered'/'forgotten' labels derived from recall_count.

    Returns the proposed w20 + the full sweep so the caller can see
    the loss landscape. Does NOT auto-apply — call
    brain_apply_fsrs_w20 to commit.
    """
    from src.fsrs_optimizer import fit_and_apply
    from src.memory import Memory
    mem = Memory()
    store, _ = await mem._ensure_initialized()
    result = fit_and_apply(
        store,
        default_w20=float(args.get("default_w20", 0.9)),
        w20_lo=float(args.get("w20_lo", 0.05)),
        w20_hi=float(args.get("w20_hi", 3.0)),
        w20_step=float(args.get("w20_step", 0.05)),
    )
    return result.to_dict()


async def handle_brain_apply_fsrs_w20(args: dict) -> dict:
    """Apply a chosen w20 to the brain's fsrs.DEFAULT_W20.

    Persists across restarts automatically: as of v0.13.0 the fsrs module
    reads DUCKBOT_FSRS_W20 at import time, so setting that env var (or
    updating it before the next process start) makes the new w20 stick.
    We also update os.environ so the running process picks it up.
    """
    import os
    from src import fsrs
    w20 = args.get("w20")
    if w20 is None:
        return {"error": "w20 is required"}
    try:
        w20 = float(w20)
    except (TypeError, ValueError):
        return {"error": f"w20 must be a number, got {w20!r}"}
    if w20 <= 0:
        return {"error": f"w20 must be > 0, got {w20}"}
    old = fsrs.DEFAULT_W20
    fsrs.DEFAULT_W20 = w20
    # Also update the env so child processes / future imports see it.
    os.environ["DUCKBOT_FSRS_W20"] = str(w20)
    return {
        "old_w20": old,
        "new_w20": w20,
        "persisted": True,
        "note": "Set in-process + DUCKBOT_FSRS_W20 env var. Persists across restarts.",
    }


async def handle_brain_fsrs_optimize_apply(args: dict) -> dict:
    """One-call: fit w20 from recall history, apply it if the MSE
    improvement vs the current default exceeds `min_improvement_pct`.
    Returns the full optimization result + whether it was applied.
    Use on a weekly cron.
    """
    from src.fsrs_optimizer import fit_and_apply
    from src.connectors.base import Brain
    from src.memory import Memory
    min_imp = float(args.get("min_improvement_pct", 1.0))
    w20_lo = float(args.get("w20_lo", 0.05))
    w20_hi = float(args.get("w20_hi", 3.0))
    w20_step = float(args.get("w20_step", 0.05))
    mem = Memory()
    store, _ = await mem._ensure_initialized()
    result = fit_and_apply(
        store,
        w20_lo=w20_lo, w20_hi=w20_hi, w20_step=w20_step,
    )
    improvement = result.baseline_mse - result.best_mse
    pct = (improvement / max(result.baseline_mse, 1e-9)) * 100
    applied = False
    old_w20 = None
    new_w20 = None
    if pct >= min_imp and result.best_w20 > 0:
        brain = Brain()
        old_w20 = brain._run_async(brain.fsrs_review_queue.__func__.__module__) if False else None  # not needed
        from src import fsrs
        old_w20 = fsrs.DEFAULT_W20
        fsrs.DEFAULT_W20 = result.best_w20
        import os
        os.environ["DUCKBOT_FSRS_W20"] = str(result.best_w20)
        new_w20 = result.best_w20
        applied = True
    return {
        "applied": applied,
        "old_w20": old_w20,
        "new_w20": new_w20,
        "improvement_pct": round(pct, 2),
        "min_required": min_imp,
        "best_w20": result.best_w20,
        "baseline_w20": result.baseline_w20,
        "n_chunks": result.n_chunks,
        "n_remembered": result.n_remembered,
        "n_forgotten": result.n_forgotten,
    }


async def handle_brain_export(args: dict) -> dict:
    """Export the entire brain as a single markdown file.

    One section per chunk, grouped by tier, with provenance. Useful for
    backups, migration to another machine, sharing with another agent,
    or inspecting the corpus in a text editor. The reverse operation
    is brain_import — together they let you round-trip the brain as
    plain markdown.
    """
    from src.tier import Tier
    from pathlib import Path
    from src.memory import Memory
    tier_filter = args.get("tier")
    if tier_filter is not None:
        tier_filter = Tier(tier_filter)
    include_super = bool(args.get("include_superseded", False))
    out_path = Path(args.get("out_path") or "data/brain_export.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mem = Memory()
    store, _ = await mem._ensure_initialized()

    lines: list[str] = ["# Brain Export", "",
                        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')}._",
                        f"_Exported by duckbot-rag-memory v{PACKAGE_VERSION}._", ""]
    per_tier: dict[str, int] = {}
    for tier_name in ("working", "episodic", "semantic", "procedural"):
        if tier_filter is not None and tier_filter.value != tier_name:
            continue
        try:
            coll = store.collection_for(Tier(tier_name))
            data = coll.get(limit=100000, include=["documents", "metadatas"])
        except Exception:
            continue
        ids = (data or {}).get("ids") or []
        docs = (data or {}).get("documents") or []
        metas = (data or {}).get("metadatas") or []
        if not ids:
            continue
        lines.append(f"## {tier_name.title()}")
        lines.append("")
        per_tier[tier_name] = 0
        for cid, doc, md in zip(ids, docs, metas):
            md = md or {}
            if (not include_super) and md.get("superseded_by"):
                continue
            src = md.get("source_path", "memory") or "memory"
            tier_label = md.get("tier", tier_name)
            imp = float(md.get("importance", 0.0) or 0.0)
            lines.append(f"### {cid}  (tier={tier_label}, importance={imp:.2f})")
            lines.append(f"_source: {src}_")
            lines.append("")
            # Body: indent each non-empty line with two spaces so the
            # downstream brain_import can detect section boundaries.
            for line in (doc or "").splitlines():
                if line.strip():
                    lines.append("  " + line)
                else:
                    lines.append("")
            lines.append("")
            per_tier[tier_name] += 1
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "out_path": str(out_path),
        "total_chunks": sum(per_tier.values()),
        "per_tier": per_tier,
        "include_superseded": include_super,
    }


async def handle_brain_import(args: dict) -> dict:
    """Import a markdown file into the brain.

    Each top-level section (## Heading) becomes one remembered chunk;
    the section text is the content. The section heading is
    auto-detected for tier:
      - 'rule' / 'preference' / 'decision' / 'setup' / 'how to' / 'always' /
        'never' / 'must' / 'should not' → procedural
      - 'YYYY-MM-DD' or 'today' / 'yesterday' → episodic
      - otherwise → semantic (general knowledge)
    Source paths are stamped so brain_recall can cite provenance.
    """
    from pathlib import Path
    from src.memory import Memory
    import re
    in_path_str = args.get("in_path")
    if not in_path_str:
        return {"error": "in_path is required"}
    in_path = Path(in_path_str)
    if not in_path.exists():
        return {"error": f"file not found: {in_path}"}
    source_path = args.get("source_path") or in_path.name
    text = in_path.read_text(encoding="utf-8", errors="replace")

    def _parse_export_chunks(raw_text: str) -> list[dict]:
        tier_re = re.compile(r"^##\s+(.+?)\s*$")
        chunk_re = re.compile(r"^###\s+(.+?)\s*$")
        meta_re = re.compile(
            r"^(?P<id>.+?)\s+\(tier=(?P<tier>[^,]+),\s*importance=(?P<importance>[0-9.]+)\)\s*$"
        )
        source_re = re.compile(r"^_source:\s*(?P<source>.+?)_\s*$")
        chunks: list[dict] = []
        current_tier = None
        current = None
        saw_chunk = False

        def flush() -> None:
            nonlocal current
            if not current:
                return
            body = "\n".join(
                line[2:] if line.startswith("  ") else line
                for line in current["body_lines"]
            ).strip()
            if body:
                current["body"] = body
                chunks.append(current)
            current = None

        for raw in raw_text.splitlines():
            tier_match = tier_re.match(raw)
            if tier_match:
                flush()
                current_tier = tier_match.group(1).strip().lower()
                continue

            chunk_match = chunk_re.match(raw)
            if chunk_match:
                flush()
                saw_chunk = True
                header = chunk_match.group(1).strip()
                title = header
                tier_hint = current_tier
                importance = None
                meta_match = meta_re.match(header)
                if meta_match:
                    title = meta_match.group("id").strip()
                    tier_hint = meta_match.group("tier").strip()
                    importance = float(meta_match.group("importance"))
                current = {
                    "title": title,
                    "tier": tier_hint,
                    "importance": importance,
                    "source_path": None,
                    "body_lines": [],
                }
                continue

            if current is not None:
                source_match = source_re.match(raw)
                if source_match and current["source_path"] is None:
                    current["source_path"] = source_match.group("source").strip()
                else:
                    current["body_lines"].append(raw)

        flush()
        return chunks if saw_chunk else []

    def _parse_generic_sections(raw_text: str) -> list[tuple[str, str]]:
        section_re = re.compile(r"(?m)^##\s+(.+?)$")
        parts = section_re.split(raw_text)
        sections: list[tuple[str, str]] = []
        for i in range(1, len(parts), 2):
            if i + 1 >= len(parts):
                break
            title = parts[i].strip()
            body = parts[i + 1].strip()
            if not body:
                continue
            body = "\n".join(
                line[2:] if line.startswith("  ") else line
                for line in body.splitlines()
            )
            sections.append((title, body))
        return sections

    parsed_chunks = _parse_export_chunks(text)

    # Heuristic tier classifier for the import.
    PROC_KEYS = ("rule", "preference", "decision", "setup", "how to",
                "always", "never", "must", "should not", "do not")
    EPIS_KEYS = ("today", "yesterday", "log", "session")
    DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")

    def _classify(title: str, body: str) -> str:
        head = (title + " " + body[:200]).lower()
        if DATE_RE.search(title) or any(k in head for k in EPIS_KEYS):
            return "episodic"
        if any(k in head for k in PROC_KEYS):
            return "procedural"
        return "semantic"

    mem = Memory()
    stored = 0
    skipped = 0
    if parsed_chunks:
        for chunk in parsed_chunks:
            title = str(chunk["title"])
            body = str(chunk["body"])
            tier = chunk["tier"] or _classify(title, body)
            source_for_chunk = chunk["source_path"] or source_path
            metadata = {
                "imported_from": str(in_path),
                "import_title": title,
                "exported_chunk_id": title,
            }
            if chunk["tier"] is not None:
                metadata["exported_tier"] = str(chunk["tier"])
            if chunk["importance"] is not None:
                metadata["exported_importance"] = float(chunk["importance"])
            if chunk["source_path"] is not None:
                metadata["exported_source_path"] = str(chunk["source_path"])
            try:
                r = await mem.remember(
                    f"# {title}\n\n{body}",
                    source_path=str(source_for_chunk),
                    force_tier=tier,
                    metadata=metadata,
                )
                if r.stored:
                    stored += 1
                else:
                    skipped += 1
            except Exception as exc:
                skipped += 1
                # continue with the next section; don't abort the whole import.
                print(f"⚠ import failed for {title!r}: {exc}", file=sys.stderr)
    else:
        sections = _parse_generic_sections(text)
        if not sections:
            return {"error": "no ## sections found in input file"}
        for title, body in sections:
            tier = _classify(title, body)
            try:
                r = await mem.remember(
                    f"# {title}\n\n{body}",
                    source_path=f"{source_path}#{title}",
                    force_tier=tier,
                    metadata={"imported_from": str(in_path), "import_title": title},
                )
                if r.stored:
                    stored += 1
                else:
                    skipped += 1
            except Exception as exc:
                skipped += 1
                # continue with the next section; don't abort the whole import.
                print(f"⚠ import failed for {title!r}: {exc}", file=sys.stderr)

    return {
        "in_path": str(in_path),
        "sections_seen": len(parsed_chunks) if parsed_chunks else len(sections),
        "stored": stored,
        "skipped": skipped,
        "source_path_stamped": source_path,
    }


async def handle_brain_seed_demo(args: dict) -> dict:
    """Seed the brain with a small bundled sample corpus.

    Useful for demos, smoke tests, and onboarding. Idempotent: skips
    chunks that already exist (matched by chunk_id which is content-hash
    based) unless --force is set.
    """
    from src.memory import Memory
    from src.tier import Tier
    import time as _time
    force = bool(args.get("force", False))
    # The demo corpus is short and curated. Each section is one
    # remembered chunk with auto-detected tier.
    DEMO = [
        ("Project: DuckBot", "DuckBot is the personal AI assistant I'm building. Stack: Python 3.12, ChromaDB, FastMCP. Currently focused on RAG + long-term memory.", "working"),
        ("Decision: local-first embeddings", "I'm using LM Studio (nomic-embed-text-v1.5) for embeddings. Decision rationale: privacy + zero API cost. Re-evaluate if recall quality drops below 0.9.", "semantic"),
        ("Rule: never commit secrets", "Always run scripts/secret-scan.sh before committing. If it fails, fix the leak. The .env file is gitignored for a reason.", "procedural"),
        ("How to restart the BATMAN container", "1. docker ps | grep batman 2. docker restart <id> 3. tail -f /var/log/batman.log. If still down, check data/chroma/ for lock files.", "procedural"),
        ("Duckets prefers dark mode", "User explicitly stated dark mode preference in 2026-05. Don't ask again. Apply to all UIs and themes going forward.", "semantic"),
        ("Project: Hermes Agent", "Hermes Agent is the self-improving agent from Nous Research. We use it as a target for our brain via the MCP stdio interface.", "working"),
    ]
    mem = Memory()
    stored = 0
    skipped = 0
    for title, body, tier in DEMO:
        # Check if a chunk with this exact content already exists.
        try:
            recall_result = await mem.recall(
                title, k=3, tier=Tier(tier), min_importance=0.0,
            )
        except Exception:
            recall_result = ([], None)
        if isinstance(recall_result, tuple):
            existing, _ = recall_result
        else:
            existing = recall_result
        # Cheap dedup: if any existing chunk's text starts with the
        # demo body, skip.
        if not force and any(
            (r.text or "").strip().startswith(body.strip()[:60])
            for r in existing
        ):
            skipped += 1
            continue
        try:
            r = await mem.remember(
                f"# {title}\n\n{body}",
                source_path="<brain_seed_demo>",
                force_tier=tier,
                metadata={"seed": "demo", "seed_title": title},
            )
            if r.stored:
                stored += 1
        except Exception as exc:
            skipped += 1
            print(f"⚠ seed failed for {title!r}: {exc}", file=sys.stderr)
    return {
        "stored": stored,
        "skipped": skipped,
        "force": force,
    }


def _check_rate_limit_or_error(tool_name: str) -> dict | None:
    """Per-tool rate-limit guard. Returns None on allowed, or a
    429-style error dict on exceeded. Dispatch loop short-circuits when
    this returns non-None. DUCKBOT_RATELIMIT_DISABLE=1 turns the check
    off entirely (returns None always).
    """
    from src.ratelimit import get_rate_limiter
    allowed, info = get_rate_limiter().check(tool_name)
    if allowed:
        return None
    return {
        "error": "rate_limited",
        "tool": tool_name,
        "limit_per_min": info.get("limit_per_min"),
        "current_tokens": info.get("current_tokens"),
        "retry_after_seconds": info.get("retry_after", 0.0),
        "message": (
            f"Rate limit exceeded for {tool_name} "
            f"({info.get('limit_per_min')}/min). "
            f"Retry in {info.get('retry_after', 0.0)}s. "
            f"Set DUCKBOT_RATELIMIT_DISABLE=1 to disable."
        ),
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
    "brain_decay_apply": handle_brain_decay_apply,
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
    "brain_skills_list": handle_brain_skills_list,
    "brain_skills_suggest": handle_brain_skills_suggest,
    "brain_skills_promote": handle_brain_skills_promote,
    "brain_user_model": handle_brain_user_model,
    "brain_palace": handle_brain_palace,
    "brain_optimize_fsrs": handle_brain_optimize_fsrs,
    "brain_apply_fsrs_w20": handle_brain_apply_fsrs_w20,
    "brain_fsrs_optimize_apply": handle_brain_fsrs_optimize_apply,
    "brain_export": handle_brain_export,
    "brain_import": handle_brain_import,
    "brain_seed_demo": handle_brain_seed_demo,
    "brain_sync": handle_brain_sync,
}


# v0.15.1: lifecycle event capture (data/events.db). Lazy-loaded so the
# rest of the module imports cleanly even if the filesystem blocks writes
# (read-only install, broken permissions, etc.).
import os as _os
from pathlib import Path as _Path
EVENT_LOG_PATH = _Path(
    _os.environ.get("DUCKBOT_EVENTS_DB") or (_Path("data") / "events.db")
)
_event_store_singleton = None


def _get_event_store():
    """Lazy-init the EventStore singleton. Returns None on failure."""
    global _event_store_singleton
    if _event_store_singleton is not None:
        return _event_store_singleton
    try:
        from src.events import EventStore as _ES
        _event_store_singleton = _ES(EVENT_LOG_PATH)
        return _event_store_singleton
    except Exception as _e:  # noqa: BLE001
        import sys as _s
        print(f"[mcp_server] event capture disabled: {_e}", file=_s.stderr)
        return None


# Event-type constants are imported lazily (see _get_event_store) but the
# capture sites inside mcp_stdio need them as names. Alias them once.
from src.events import (  # noqa: E402  -- imported for side-effect aliasing
    SESSION_START,
    SESSION_END,
    PRE_TOOL_USE,
    POST_TOOL_USE,
    TOOL_ERROR,
)

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
    import time as _time
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

    # v0.15.1: lifecycle event capture. The EventStore is optional —
    # if events can't be initialized (read-only filesystem, etc.) we
    # log + continue without events rather than failing the server.
    _event_store = _get_event_store()
    _current_session_id: str = ""
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
                    params = msg.get("params", {})
                    # v0.15.1: extract a session_id from the initialize
                    # params so events can be grouped per-MCP-session.
                    # Prefer explicit _meta.session_id (MCP convention),
                    # fall back to clientInfo.name, then a UUID-like id.
                    _client_info = params.get("clientInfo") or {}
                    _init_meta = params.get("_meta") or {}
                    _current_session_id = (
                        _init_meta.get("session_id")
                        or _client_info.get("session_id")
                        or _client_info.get("name")
                        or f"mcp-{_os.getpid()}-{int(_time.time())}"
                    )
                    if _event_store is not None:
                        _event_store.record_event(
                            _current_session_id,
                            SESSION_START,
                            context={
                                "client": _client_info.get("name"),
                                "version": _client_info.get("version"),
                                "platform": _init_meta.get("platform"),
                            },
                        )
                    resp = {
                        "jsonrpc": "2.0", "id": mid,
                        "result": {
                            "protocolVersion": "2024-11-05",
                            "serverInfo": {"name": "duckbot-memory", "version": PACKAGE_VERSION},
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
                        # Rate-limit check (per-tool token bucket). DUCKBOT_RATELIMIT_DISABLE=1
                        # bypasses. Returns a 429-style error before invoking
                        # the handler so a misbehaving agent can't bypass
                        # the limit by retrying.
                        rl_err = _check_rate_limit_or_error(name)
                        if rl_err is not None:
                            resp = {
                                "jsonrpc": "2.0", "id": mid,
                                "error": {
                                    "code": -32029,  # -32029 = "Server rate limit" (JSON-RPC)
                                    "message": rl_err.get("message"),
                                    "data": rl_err,
                                },
                            }
                        else:
                            # v0.15.1: pre/post event capture. Each tool
                            # call logs PRE (with args) + POST (with
                            # result + duration) or TOOL_ERROR (with
                            # exception message). Events are best-effort:
                            # if the EventStore raises, we still return
                            # the tool's result.
                            _call_started = _time.monotonic()
                            _event_store = _get_event_store()
                            if _event_store is not None:
                                try:
                                    _event_store.record_event(
                                        _current_session_id,
                                        PRE_TOOL_USE,
                                        tool_name=name,
                                        args=args,
                                    )
                                except Exception:  # noqa: BLE001
                                    pass
                            try:
                                result = _server_loop.run_until_complete(handler(args))
                                resp = {
                                    "jsonrpc": "2.0", "id": mid,
                                    "result": {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]},
                                }
                                if _event_store is not None:
                                    try:
                                        _event_store.record_event(
                                            _current_session_id,
                                            POST_TOOL_USE,
                                            tool_name=name,
                                            result=result,
                                            duration_ms=int((_time.monotonic() - _call_started) * 1000),
                                        )
                                    except Exception:  # noqa: BLE001
                                        pass
                            except Exception as exc:
                                resp = {"jsonrpc": "2.0", "id": mid, "error": {"code": -32603, "message": str(exc)}}
                                if _event_store is not None:
                                    try:
                                        _event_store.record_event(
                                            _current_session_id,
                                            TOOL_ERROR,
                                            tool_name=name,
                                            error=str(exc),
                                            duration_ms=int((_time.monotonic() - _call_started) * 1000),
                                        )
                                    except Exception:  # noqa: BLE001
                                        pass
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
