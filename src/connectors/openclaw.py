"""
connectors/openclaw.py - OpenClaw integration for the DuckBot brain.

.. deprecated:: v0.14.0
    Prefer `src.extensions.duckbot_brain.adapter` (the stdio JSON-RPC
    OpenClaw extension, 12-tool core agent surface) or
    `src.connectors.openclaw_shim` (the shell CLI shim). This module is
    retained for backwards compatibility with deployments that wired up
    `python -m src.connectors.openclaw` directly. It will not be removed
    (per the project's "No deletions" rule in AGENTS.md), but new code
    should not import from here.

Two pieces (legacy):

  1. TOOL_DEFINITIONS - the MCP tool schema (name, description, inputSchema)
     that OpenClaw's MCP client will surface to the model. Covers all 5 layers:
     vector (remember/recall/reflect/forget/stats), graph (3 tools), blocks
     (5 tools), quarantine (3 tools), and dashboard (1 tool).

  2. handle() - dispatcher that runs a tool call against the Brain facade and
     returns a JSON-serializable result. Stays pure-Python / stdlib so it can
     be hosted in `python -m src.mcp_server` without any framework dependency.

OpenClaw wires this up by adding to `~/.openclaw/openclaw.json`:

  {
    "mcp": {
      "servers": {
        "duckbot-brain": {
          "command": "<repo>/.venv/bin/python",
          "args": ["-m", "src.mcp_server"],
          "cwd": "<repo>",
          "env": {
            "DUCKBOT_EMBEDDING": "minimax",
            "MINIMAX_API_KEY": "<redacted - read from openclaw.json secret store>"
          }
        }
      }
    }
  }

Restart OpenClaw gateway to pick up the new MCP server, then `mcporter list`
should show 27 tools (this file's 27 brain_* tools — OpenClaw has its own
non-overlapping stack for the base `remember`/`recall`/`reflect`/etc.).

No LLM, no paid services, no surprises. Local stdlib only.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .base import Brain, _run_async


# -----------------------------------------------------------------------------
# MCP tool schema
# -----------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict] = [
    # ----- Vector store (mirror of mcp_server.py; included here for completeness) -----
    {
        "name": "brain_stats",
        "description": "One-glance snapshot of all 5 brain layers: vector, graph, blocks, quarantine.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "brain_remember",
        "description": "Save a memory. Pre-scanned for injection — suspicious text is quarantined, not stored.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "the memory content"},
                "source_path": {"type": "string", "description": "where this came from"},
                "force_tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"]},
                "skip_scan": {"type": "boolean", "default": False, "description": "skip injection scan (use only for trusted callers)"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "brain_recall_verbatim",
        "description": "Verbatim-first recall. Returns chunks with their exact text preserved.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 5},
                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"]},
                "min_importance": {"type": "number"},
                "rerank": {"type": "boolean", "default": False},
                "decay": {"type": "boolean", "default": False},
            },
            "required": ["query"],
        },
    },
    {
        "name": "brain_recall",
        "description": "Hybrid retrieval over all chunks. Returns top-k with tier, source, importance, score.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 5},
                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"]},
                "min_importance": {"type": "number"},
                "rerank": {"type": "boolean", "default": False, "description": "Layer 7: run cross-encoder rerank (qwen3-reranker-0.6b). Default off; pass true to opt in. No paid API."},
                "decay": {"type": "boolean", "default": False, "description": "Layer 8: apply Ebbinghaus retention weighting. Default off; pass true to opt in. Public-domain math (1885), no LLM call. Boosts recently-recalled chunks."},
                "tier_priors": {"type": "boolean", "default": False, "description": "Layer 11: apply per-tier multiplicative priors (procedural=1.5, semantic=1.2, episodic=1.0, working=0.8). Default off; pass true to opt in. Boosts procedural rules and demotes working chatter."},
                "tier_priors_overrides": {"type": "object", "description": "Optional per-tier prior overrides, e.g. {\"procedural\": 2.0}. Tier names not in the dict fall back to defaults."},
                "fsrs": {"type": "boolean", "default": False, "description": "Layer 9: apply FSRS-6 power-law retrievability weighting. Default off; pass true to opt in. Uses per-chunk stability_days + difficulty from metadata. Public-domain algorithm spec."},
            },
            "required": ["query"],
        },
    },
    {
        "name": "brain_reflect",
        "description": "Consolidate episodic chunks into semantic memory.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "lookback_days": {"type": "integer", "default": 7},
                "max_chunks": {"type": "integer", "default": 200},
            },
        },
    },

    # ----- Knowledge graph (Layer 1) -----
    {
        "name": "brain_graph_entity",
        "description": "Create or update a knowledge-graph entity by name and kind.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "kind": {"type": "string", "default": "concept", "description": "person|project|file|place|concept|fact"},
                "properties": {"type": "object"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "brain_graph_relate",
        "description": "Add a relationship between two graph entities with a label.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "source entity name"},
                "target": {"type": "string", "description": "target entity name"},
                "label": {"type": "string", "description": "e.g. works_on, uses, depends_on, replaced, rotated"},
                "properties": {"type": "object"},
            },
            "required": ["source", "target", "label"],
        },
    },
    {
        "name": "brain_graph_query",
        "description": "Query the knowledge graph for entities by name or kind.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "substring match (optional)"},
                "kind": {"type": "string", "description": "filter by kind (optional)"},
            },
        },
    },
    {
        "name": "brain_graph_relationships",
        "description": "List all relationships for an entity, optionally at a specific time.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity": {"type": "string"},
                "at": {"type": "number", "description": "unix epoch seconds; default = now"},
            },
            "required": ["entity"],
        },
    },
    {
        "name": "brain_graph_history",
        "description": "Show the history of changes for an entity over time.",
        "inputSchema": {
            "type": "object",
            "properties": {"entity": {"type": "string"}},
            "required": ["entity"],
        },
    },
    {
        "name": "brain_graph_precursors",
        "description": (
            "Causal precursor tracing (Observer Perspective). Walks the "
            "graph backward from an entity through decided_by / depends_on "
            "/ learned_from / caused_by / supports edges to surface WHY we "
            "know something. Returns a depth-indexed chain + critical_depth "
            "(shallowest depth capturing >=90% of influence) + coverage "
            "(fraction of immediate edges with upstream rationale). Use "
            "before making a decision to understand the full reasoning "
            "chain. Inspired by MindBank's Observer Perspective."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity": {"type": "string", "description": "entity to trace backward from (e.g. 'Use Postgres')"},
                "max_depth": {"type": "integer", "default": 3, "minimum": 1, "maximum": 10,
                              "description": "BFS depth limit (default 3 hops)"},
                "include_inactive": {"type": "boolean", "default": False,
                                     "description": "include ended relationships"},
                "min_influence": {"type": "number", "default": 0.0,
                                  "description": "drop precursors whose decayed influence is below this floor"},
            },
            "required": ["entity"],
        },
    },
    {
        "name": "brain_graph_blind_spots",
        "description": (
            "Identify orphan decisions in the graph — entities that have "
            "outgoing causal edges (decided_by / depends_on / learned_from / "
            "etc.) but no upstream rationale of their own. Severity scales "
            "with downstream edge count: 1 = low, 2 = medium, 3+ = high. Use "
            "to surface facts the agent believes but can't explain WHY."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "max_results": {"type": "integer", "default": 50, "minimum": 1, "maximum": 500,
                                "description": "cap on returned blind spots"},
                "include_inactive": {"type": "boolean", "default": False,
                                     "description": "include entities whose relationships are inactive"},
            },
        },
    },
    {
        "name": "brain_graph_cognify",
        "description": "Cognee ECL stage 2: dedupe + reconcile entity relations. Finds duplicate (source, target, label) triples + duplicate aliases. Default dry-run. Use to clean up a graph that's accumulated duplicates over time.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "dry_run": {"type": "boolean", "default": True, "description": "preview only — don't actually merge"},
            },
        },
    },
    {
        "name": "brain_graph_reconcile",
        "description": "Cognee ECL stage 3: typed-schema reconcile. Deletes orphan relationships (source/target entity missing) + self-loops. Always writes (no dry-run).",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "brain_inspect",
        "description": "Consolidated entity view: graph + recent memories + blocks. Given an entity name, returns everything the brain knows about it in one dict. Useful for audit + agent self-inspection: 'what does the brain actually know about X?'",
        "inputSchema": {
            "type": "object",
            "properties": {
                "entity": {"type": "string", "description": "entity name (e.g. 'Duckets', 'OpenClaw', 'BATMAN')"},
                "k": {"type": "integer", "default": 10, "description": "max memories to recall"},
            },
            "required": ["entity"],
        },
    },

    # ----- Memory blocks (Layer 3) -----
    {
        "name": "brain_block_read",
        "description": "Read a memory block (a self-editing persona/rule container) by name.",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "brain_block_write",
        "description": "Create or overwrite a memory block with the given text.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["name", "text"],
        },
    },
    {
        "name": "brain_block_append",
        "description": "Append text to an existing memory block.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "text": {"type": "string"},
            },
            "required": ["name", "text"],
        },
    },
    {
        "name": "brain_block_delete",
        "description": "Delete a memory block by name.",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "brain_block_list",
        "description": "List all memory blocks with their character counts and last-modified times.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "brain_seed_blocks",
        "description": "Seed the default memory blocks (persona, user, today_focus) if missing.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "brain_seed_demo",
        "description": "Seed a small demo memory (idempotent — skips if already present unless force=true). Useful for verifying the brain round-trip end-to-end.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "force": {"type": "boolean", "default": False, "description": "re-store even if already present"},
            },
        },
    },

    # ----- Injection scan (Layer 4) -----
    {
        "name": "brain_injection_scan",
        "description": "Scan text for prompt-injection patterns. Returns verdict and reasons.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "brain_quarantine_list",
        "description": "List quarantined items by status (pending, approved, rejected).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["pending", "approved", "rejected", "all"], "default": "pending"},
            },
        },
    },
    {
        "name": "brain_quarantine_review",
        "description": "Review a quarantined item: approve (re-store) or discard.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scan_id": {"type": "string"},
                "decision": {"type": "string", "enum": ["approved", "rejected"]},
                "reviewer": {"type": "string", "default": "operator"},
            },
            "required": ["scan_id", "decision"],
        },
    },

    # ----- v0.10.0 — useful MCP tools extension -----
    {
        "name": "brain_fsrs_review",
        "description": "Get the FSRS-6 spaced-repetition review queue.",
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
        "description": "Show retention/decay status for recent chunks.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"]},
                "k": {"type": "integer", "default": 50},
            },
        },
    },
    {
        "name": "brain_forget_by_query",
        "description": "Delete the top-k chunks matching a query. Destructive.",
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
        "name": "brain_search_verbatim",
        "description": "Exact substring match against stored verbatim text.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "needle": {"type": "string"},
                "k": {"type": "integer", "default": 5},
            },
            "required": ["needle"],
        },
    },
    # ----- v0.11.0 — OpenClaw dreaming bridge + Hermes /learn + Active Memory -----
    {
        "name": "brain_dreaming_read",
        "description": "Pull dream entries from OpenClaw dreaming surface into the brain.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "brain_dreaming_cycle",
        "description": "Distill high-importance chunks into a new dream entry.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "k": {"type": "integer", "default": 10},
                "min_importance": {"type": "number", "default": 0.5},
            },
        },
    },
    {
        "name": "brain_learn",
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
        "name": "brain_active_memory",
        "description": "Dispatch an Active Memory tool call.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool": {"type": "string", "enum": ["memory_query", "memory_store", "memory_recent", "memory_forget"]},
                "args": {"type": "object"},
            },
            "required": ["tool"],
        },
    },
]  


# -----------------------------------------------------------------------------
# Tool dispatcher
# -----------------------------------------------------------------------------

def _serialize(obj: Any) -> Any:
    """Coerce dataclass / Path / etc. to JSON-safe primitives."""
    if obj is None:
        return None
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return obj.to_dict()
    if isinstance(obj, list):
        return [_serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


_VALID_TIERS = ("working", "episodic", "semantic", "procedural")


def _validate_tier(args: dict, tool_name: str) -> dict | None:
    """Return an error dict if tier is invalid, else None.

    brain.recall(..., tier="invalid") raises ValueError("'invalid' is not
    a valid Tier") which propagates through _run_async and surfaces to
    the MCP client as a generic -32603 error. Pre-validating here gives
    a clear "tier must be one of ..." message instead.
    """
    tier = args.get("tier")
    if tier is not None and tier not in _VALID_TIERS:
        return {"error": f"tier must be one of {list(_VALID_TIERS)}, got {tier!r}", "tool": tool_name}
    return None


    {
        "name": "brain_wake_up",
        "description": "One-call session-start context load: recent memories + active blocks + graph summary + FSRS queue + stats.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "optional anchor query"},
                "k": {"type": "integer", "default": 8},
                "include_blocks": {"type": "boolean", "default": True},
                "include_graph": {"type": "boolean", "default": True},
                "include_fsrs_review": {"type": "boolean", "default": True},
            },
        },
    },
    {
        "name": "brain_inflate",
        "description": "Recall relevant memories and format them as a markdown context block for direct agent injection.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 10},
                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"]},
                "min_importance": {"type": "number", "default": 0.3},
            },
            "required": ["query"],
        },
    },
    {
        "name": "brain_nudge",
        "description": "Proactive memory nudge: surface stale-but-important memories the agent might be forgetting.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "context": {"type": "string", "description": "optional current focus"},
                "k": {"type": "integer", "default": 5},
                "min_importance": {"type": "number", "default": 0.6},
                "stale_days": {"type": "integer", "default": 7},
            },
        },
    },
    {
        "name": "brain_palace",
        "description": "Wing/Room/Drawer 2D view of the brain. No args: list wings. With --wing: walk that wing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "wing": {"type": "string"},
                "room": {"type": "string"},
                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"]},
            },
        },
    },
    {
        "name": "brain_user_model",
        "description": "Aggregate user-related facts into a single 'user' memory block. Honcho-inspired.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "block_name": {"type": "string", "default": "user"},
                "min_importance": {"type": "number", "default": 0.5},
                "max_facts": {"type": "integer", "default": 30},
                "k_per_query": {"type": "integer", "default": 50},
            },
        },
    },
    {
        "name": "brain_export",
        "description": "Export the brain to a markdown file (per-tier sections with metadata). Idempotent.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "out_path": {"type": "string"},
                "include_superseded": {"type": "boolean", "default": False},
            },
            "required": ["out_path"],
        },
    },
    {
        "name": "brain_import",
        "description": "Import a markdown export file into the brain. Auto-detects tier from section headings.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "in_path": {"type": "string"},
                "source_path": {"type": "string"},
            },
            "required": ["in_path"],
        },
    },
    {
        "name": "brain_index",
        "description": "Compact whole-corpus summary using the AAAK dialect. LLM scans in <500 tokens, picks entries to expand.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"]},
                "max_chunks": {"type": "integer", "default": 5000},
                "preview_chars": {"type": "integer", "default": 80},
            },
        },
    },
    {
        "name": "brain_sync",
        "description": "Sync stored memories to OpenClaw/Hermes context files (MEMORY.md, USER.md, SOUL.md).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "enum": ["openclaw", "hermes", "both"], "default": "both"},
                "memory_k": {"type": "integer", "default": 20},
                "user_k": {"type": "integer", "default": 5},
                "dry_run": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "brain_skill_create",
        "description": "Auto-generate an agentskills.io SKILL.md from a task description + instructions. Writes skills/<slug>/SKILL.md.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "description": {"type": "string"},
                "instructions": {"type": "array", "items": {"type": "string"}},
                "example": {"type": "string"},
                "emoji": {"type": "string"},
                "overwrite": {"type": "boolean", "default": False},
            },
            "required": ["name", "description", "instructions"],
        },
    },
    {
        "name": "brain_skills_list",
        "description": "List unpromoted skill-candidate chunks. Agent reads these and decides which to promote.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "include_promoted": {"type": "boolean", "default": False},
                "k": {"type": "integer", "default": 50},
            },
        },
    },
    {
        "name": "brain_skills_promote",
        "description": "Promote a skill candidate to a full SKILL.md. AGENT authors the content (name/description/instructions or instructions_markdown). Pure template, no LLM.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "chunk_id": {"type": "string"},
                "name": {"type": "string"},
                "description": {"type": "string"},
                "instructions": {"type": "array", "items": {"type": "string"}},
                "instructions_markdown": {"type": "string"},
                "example": {"type": "string"},
                "emoji": {"type": "string"},
                "overwrite": {"type": "boolean", "default": False},
            },
            "required": ["chunk_id", "name", "description"],
        },
    },
    {
        "name": "brain_skills_suggest",
        "description": "Semantic top-N skill candidates matching a query (agent-driven pipeline). No LLM.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    # v0.15.0: skills-aware brain tools (all delegate to the canonical MCP
    # handlers so the connector and the MCP server stay in lock-step).
    {
        "name": "brain_apply_fsrs_w20",
        "description": "Apply a chosen w20 (persists via DUCKBOT_FSRS_W20 env var). Run after optimize-fsrs picks a value.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "w20": {"type": "number"},
            },
            "required": ["w20"],
        },
    },
    {
        "name": "brain_fsrs_optimize_apply",
        "description": "Optimize then auto-apply if the new w20 is better than the baseline.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "default_w20": {"type": "number", "default": 0.9},
                "w20_lo": {"type": "number", "default": 0.1},
                "w20_hi": {"type": "number", "default": 1.5},
                "w20_step": {"type": "number", "default": 0.05},
                "min_improvement_pct": {"type": "number", "default": 1.0},
            },
        },
    },
    {
        "name": "brain_optimize_fsrs",
        "description": "Self-tune the FSRS-6 forgetting-curve exponent. Returns proposed w20 + sweep.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "default_w20": {"type": "number", "default": 0.9},
                "w20_lo": {"type": "number", "default": 0.1},
                "w20_hi": {"type": "number", "default": 1.5},
                "w20_step": {"type": "number", "default": 0.05},
            },
        },
    },


def handle(tool_name: str, args: dict) -> dict:
    """
    Dispatch a single MCP tool call to the Brain facade.

    Returns a JSON-serializable dict. Errors are caught and returned as
    {"error": "..."} so the MCP client can surface them cleanly.
    """
    # Map legacy unprefixed names to their brain_-prefixed handlers.
    # Clients that use the short names (remember, recall, etc.) get
    # routed to the canonical handlers. Done by rewriting tool_name
    # BEFORE the if/elif chain so the brain_*-prefixed handlers
    # below see the canonical name.
    _legacy_alias = {
        "remember": "brain_remember",
        "recall": "brain_recall",
        "recall_verbatim": "brain_recall_verbatim",
        "reflect": "brain_reflect",
        "forget": "brain_forget_by_query",
        "forget_by_query": "brain_forget_by_query",
        "stats": "brain_stats",
        "fsrs_review": "brain_fsrs_review",
        "decay_status": "brain_decay_status",
        "brain_decay_apply": "brain_decay_apply",
        "learn": "brain_learn",
        "search_verbatim": "brain_search_verbatim",
        "watch": "brain_wake_up",
        "doctor": "brain_wake_up",
        "active_memory": "brain_active_memory",
    }
    tool_name = _legacy_alias.get(tool_name, tool_name)

    try:
        brain = Brain()
        if tool_name == "brain_stats":
            return _serialize(brain.stats())
        if tool_name == "brain_remember":
            # Validate required args — previously raised KeyError on empty/missing.
            # (The canonical `remember` handler in src/mcp_server.py has richer
            # validation + skill_candidate support; this is a back-compat alias.)
            if not args.get("text") or not str(args["text"]).strip():
                return {"error": "text must be a non-empty string", "tool": tool_name}
            # Forward agent-provided facts (OpenClaw agent does the extraction).
            facts = args.get("facts")
            if facts is not None and not isinstance(facts, list):
                return {"error": "facts must be a list of strings", "tool": tool_name}
            if facts:
                facts = [f for f in facts if isinstance(f, str) and f.strip()]
            r = brain.remember(
                text=args["text"],
                source_path=args.get("source_path", "<openclaw>"),
                force_tier=args.get("force_tier"),
                skip_scan=args.get("skip_scan", False),
                facts=facts or None,
            )
            return _serialize(r)
        if tool_name == "brain_recall_verbatim":
            query = (args.get("query") or "").strip()
            if not query:
                return {"error": "query is required", "tool": tool_name}
            tier_err = _validate_tier(args, tool_name)
            if tier_err is not None:
                return tier_err
            # Validate tier_priors_overrides type — non-dict would be
            # silently ignored by maybe_apply_tier_priors, masking the
            # user's intent. Surface as a clear error instead.
            tpo = args.get("tier_priors_overrides")
            if tpo is not None and not isinstance(tpo, dict):
                return {"error": "tier_priors_overrides must be a dict", "tool": tool_name}
            results = brain.recall_verbatim(
                query=query,
                k=args.get("k", 5),
                tier=args.get("tier"),
                min_importance=args.get("min_importance"),
                rerank=args.get("rerank"),
                decay=args.get("decay"),
                tier_priors=args.get("tier_priors"),
                tier_priors_overrides=tpo,
                fsrs=args.get("fsrs"),
            )
            # Convert VerbatimResult dataclasses to dicts so the MCP
            # server's json.dumps can serialize the response. (The native
            # mcp_server.py recall_verbatim handler has the same issue.)
            return {"results": [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in results]}
        if tool_name == "brain_recall":
            query = (args.get("query") or "").strip()
            if not query:
                return {"error": "query is required", "tool": tool_name}
            tier_err = _validate_tier(args, tool_name)
            if tier_err is not None:
                return tier_err
            # Validate tier_priors_overrides type — non-dict would be
            # silently ignored (see brain_recall_verbatim fix).
            tpo = args.get("tier_priors_overrides")
            if tpo is not None and not isinstance(tpo, dict):
                return {"error": "tier_priors_overrides must be a dict", "tool": tool_name}
            results = brain.recall(
                query=query,
                k=args.get("k", 5),
                tier=args.get("tier"),
                min_importance=args.get("min_importance"),
                rerank=args.get("rerank"),
                decay=args.get("decay"),
                tier_priors=args.get("tier_priors"),
                tier_priors_overrides=tpo,
                fsrs=args.get("fsrs"),
            )
            return {"results": _serialize(results)}
        if tool_name == "brain_reflect":
            from src.memory import Memory
            # Use _run_async so this is safe from inside the MCP server's
            # running event loop (raw asyncio.run would raise RuntimeError).
            return _run_async(Memory().reflect(
                lookback_days=args.get("lookback_days", 7),
                max_chunks=args.get("max_chunks", 200),
            ))

        # Graph
        if tool_name == "brain_graph_entity":
            if not args.get("name"):
                return {"error": "name is required", "tool": tool_name}
            return brain.graph_upsert_entity(
                name=args["name"],
                kind=args.get("kind", "concept"),
                properties=args.get("properties"),
            )
        if tool_name == "brain_graph_relate":
            missing = [k for k in ("source", "target", "label") if not args.get(k)]
            if missing:
                return {"error": f"missing required argument(s): {', '.join(missing)}", "tool": tool_name}
            return brain.graph_add_relationship(
                source=args["source"],
                target=args["target"],
                label=args["label"],
                properties=args.get("properties"),
            )
        if tool_name == "brain_graph_query":
            return {"entities": brain.graph_query(name=args.get("name"), kind=args.get("kind"))}
        if tool_name == "brain_graph_relationships":
            if not args.get("entity"):
                return {"error": "entity is required", "tool": tool_name}
            return {"relationships": brain.graph_relationships(
                entity_name=args["entity"], at=args.get("at")
            )}
        if tool_name == "brain_graph_history":
            if not args.get("entity"):
                return {"error": "entity is required", "tool": tool_name}
            return {"history": brain.graph_history(entity_name=args["entity"])}

        if tool_name == "brain_graph_precursors":
            if not args.get("entity"):
                return {"error": "entity is required", "tool": tool_name}
            return brain.graph_precursors(
                entity_name=args["entity"],
                max_depth=int(args.get("max_depth", 3)),
                include_inactive=bool(args.get("include_inactive", False)),
                min_influence=float(args.get("min_influence", 0.0)),
            )

        if tool_name == "brain_graph_blind_spots":
            return brain.graph_blind_spots(
                max_results=int(args.get("max_results", 50)),
                include_inactive=bool(args.get("include_inactive", False)),
            )

        if tool_name == "brain_graph_cognify":
            return brain.graph_cognify(dry_run=bool(args.get("dry_run", True)))
        if tool_name == "brain_graph_reconcile":
            return brain.graph_reconcile()

        if tool_name == "brain_inspect":
            entity = args.get("entity")
            if not entity:
                return {"error": "entity is required"}
            return brain.inspect(entity=entity, k=int(args.get("k", 10)))

        # Blocks
        if tool_name == "brain_block_read":
            if not args.get("name"):
                return {"error": "name is required", "tool": tool_name}
            r = brain.block_read(args["name"])
            return r if r is not None else {"error": f"block not found: {args['name']}", "tool": tool_name}
        if tool_name == "brain_block_write":
            if not args.get("name") or not args.get("text"):
                return {"error": "name and text are required", "tool": tool_name}
            return brain.block_write(args["name"], args["text"])
        if tool_name == "brain_block_append":
            if not args.get("name") or not args.get("text"):
                return {"error": "name and text are required", "tool": tool_name}
            return brain.block_append(args["name"], args["text"])
        if tool_name == "brain_block_delete":
            if not args.get("name"):
                return {"error": "name is required", "tool": tool_name}
            return brain.block_delete(args["name"])
        if tool_name == "brain_block_list":
            return {"blocks": brain.block_list()}
        if tool_name == "brain_seed_blocks":
            return {"seeded": brain.seed_default_blocks()}
        if tool_name == "brain_seed_demo":
            # Delegate to the canonical MCP handler which inlines the
            # curated demo corpus (the Brain class doesn't have a seed_demo
            # method). Same as-run semantics as the MCP server.
            import asyncio
            from src.mcp_server import handle_brain_seed_demo
            return asyncio.run(handle_brain_seed_demo(args))

        # Quarantine
        if tool_name == "brain_injection_scan":
            if not args.get("text"):
                return {"error": "text is required", "tool": tool_name}
            return brain.injection_scan(args["text"])
        if tool_name == "brain_quarantine_list":
            return {"quarantined": brain.quarantine_list(status=args.get("status", "pending"))}
        if tool_name == "brain_quarantine_review":
            if not args.get("scan_id") or not args.get("decision"):
                return {"error": "scan_id and decision are required", "tool": tool_name}
            return brain.quarantine_review(
                scan_id=args["scan_id"],
                decision=args["decision"],
                reviewer=args.get("reviewer", "operator"),
            )

        # v0.10.0 — useful MCP tools extension
        if tool_name == "brain_fsrs_review":
            return {"queue": brain.fsrs_review_queue(
                tier=args.get("tier"),
                k=args.get("k", 10),
            )}
        if tool_name == "brain_decay_status":
            return brain.decay_status(tier=args.get("tier"), k=args.get("k", 50))
        if tool_name == "brain_forget_by_query":
            query = (args.get("query") or "").strip()
            if not query:
                return {"error": "query is required", "tool": tool_name}
            return brain.forget_by_query(
                query=query,
                k=args.get("k", 5),
                tier=args.get("tier"),
            )
        if tool_name == "brain_search_verbatim":
            needle = (args.get("needle") or "").strip()
            if not needle:
                return {"error": "needle must be a non-empty string", "tool": tool_name}
            return {"matches": brain.search_verbatim(
                needle=needle,
                k=args.get("k", 5),
            )}

        # v0.11.0 — OpenClaw dreaming bridge + Hermes /learn + Active Memory
        if tool_name == "brain_dreaming_read":
            return brain.dreaming_read()
        if tool_name == "brain_dreaming_cycle":
            return brain.dreaming_cycle(
                k=args.get("k", 10),
                min_importance=args.get("min_importance", 0.5),
            )
        if tool_name == "brain_learn":
            text = args.get("text") or ""
            if not text.strip():
                return {"error": "text must be a non-empty string", "tool": tool_name}
            return brain.learn(
                text=text,
                force_tier=args.get("force_tier", "procedural"),
                source=args.get("source", "<hermes-/learn>"),
                metadata=args.get("metadata"),
                invoke_hermes=args.get("invoke_hermes", True),
            )
        if tool_name == "brain_active_memory":
            if not args.get("tool"):
                return {"error": "tool is required", "tool": tool_name}
            return brain.active_memory(
                tool=args["tool"],
                args=args.get("args", {}),
            )

        # v0.15.0: delegate newer MCP tools to their canonical handlers.
        # The connector is a subset of the MCP surface; rather than
        # duplicating logic, route these to src.mcp_server.handle_*.
        import asyncio as _asyncio
        from src import mcp_server as _mcp
        _delegated = {
            "brain_wake_up": _mcp.handle_brain_wake_up,
            "brain_inflate": _mcp.handle_brain_inflate,
            "brain_nudge": _mcp.handle_brain_nudge,
            "brain_palace": _mcp.handle_brain_palace,
            "brain_user_model": _mcp.handle_brain_user_model,
            "brain_export": _mcp.handle_brain_export,
            "brain_import": _mcp.handle_brain_import,
            "brain_index": _mcp.handle_brain_index,
            "brain_sync": _mcp.handle_brain_sync,
            "brain_skill_create": _mcp.handle_brain_skill_create,
            "brain_skills_list": _mcp.handle_brain_skills_list,
            "brain_skills_promote": _mcp.handle_brain_skills_promote,
            "brain_skills_suggest": _mcp.handle_brain_skills_suggest,
            "brain_apply_fsrs_w20": _mcp.handle_brain_apply_fsrs_w20,
            "brain_fsrs_optimize_apply": _mcp.handle_brain_fsrs_optimize_apply,
            "brain_optimize_fsrs": _mcp.handle_brain_optimize_fsrs,
        }
        if tool_name in _delegated:
            try:
                return _asyncio.run(_delegated[tool_name](args))
            except Exception as exc:
                return {"error": f"{type(exc).__name__}: {exc}", "tool": tool_name}

        return {"error": f"unknown tool: {tool_name}"}
    except Exception as e:
        # Do NOT include `args` in the response — it may contain the user's
        # raw prompt (including any prompt-injection text that triggered
        # the quarantine path), and we'd be echoing attacker-controlled
        # bytes back through the MCP client. Log the full args server-side
        # if needed; surface only the error class + message.
        import logging
        logging.getLogger(__name__).warning(
            "openclaw tool %s failed: %s", tool_name, e, exc_info=True,
        )
        return {"error": f"{type(e).__name__}: {e}", "tool": tool_name}


# -----------------------------------------------------------------------------
# OpenClaw config snippet (for documentation)
# -----------------------------------------------------------------------------

def openclaw_config_snippet(repo_path: str = "/Users/duckets/Desktop/duckbot-rag-memory") -> dict:
    """
    Return the JSON snippet Duckets should merge into ~/.openclaw/openclaw.json
    under "mcp.servers.duckbot-brain".

    This is a helper, NOT auto-applied. Operator must review and merge manually.
    """
    return {
        "duckbot-brain": {
            "command": f"{repo_path}/.venv/bin/python",
            "args": ["-m", "src.mcp_server"],
            "cwd": repo_path,
            "description": "DuckBot brain: vector + graph + blocks + quarantine + dreaming + /learn + active-memory (27 tools)",
            "env": {
                "DUCKBOT_EMBEDDING": "minimax",
                # MINIMAX_API_KEY should be read from openclaw.json's secrets,
                # not hardcoded. See openclaw.json docs for the secret-store format.
            },
        }
    }


# -----------------------------------------------------------------------------
# CLI self-test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Print a status report: tool count + config snippet
    print(f"OpenClaw connector: {len(TOOL_DEFINITIONS)} MCP tools defined")
    print()
    print("To wire into OpenClaw, add to ~/.openclaw/openclaw.json under 'mcp.servers':")
    print(json.dumps(openclaw_config_snippet(), indent=2))
    print()
    print("Then `openclaw gateway restart` and `mcporter list` to verify.")
    print()
    print("Live stats (real path):")
    print(json.dumps(_serialize(Brain().stats()).__str__()[:200] + "..."))
