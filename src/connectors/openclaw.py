"""
connectors/openclaw.py - OpenClaw integration for the DuckBot brain.

Two pieces:

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
should show 16 tools (the 7 from `src.mcp_server.py` + 9 new from this file).

No LLM, no paid services, no surprises. Local stdlib only.
"""

from __future__ import annotations

import json
import time
from typing import Any

from .base import Brain


# -----------------------------------------------------------------------------
# MCP tool schema
# -----------------------------------------------------------------------------

TOOL_DEFINITIONS: list[dict] = [
    # ----- Vector store (mirror of mcp_server.py; included here for completeness) -----
    {
        "name": "brain_stats",
        "description": "One-glance snapshot of all 5 brain layers: vector store, knowledge graph, memory blocks, quarantine.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "brain_remember",
        "description": "Save a memory. Auto-chunks, classifies tier, extracts entities, embeds, stores. Pre-scanned for injection - suspicious text is quarantined, not stored.",
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
        "description": "Layer 13 verbatim-first retrieval: returns the original (pre-overlap, pre-prefix) source text instead of the contextualized chunk. Use when the user asks 'what exactly did I say about X?' so we never paraphrase or summarize.",
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
        "description": "Hybrid retrieval (vector + BM25 + RRF, plus optional cross-encoder rerank, Ebbinghaus decay, tier priors, and FSRS-6 spaced repetition) over all chunks. Returns top-k with tier, source, importance, score.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "default": 5},
                "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"]},
                "min_importance": {"type": "number"},
                "rerank": {"type": "boolean", "default": False, "description": "Layer 7: run cross-encoder rerank (BGE bge-reranker-base). Default off; pass true to opt in. No paid API."},
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
        "description": "Sleep-time consolidation: episodic → semantic distillation.",
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
        "description": "Add or update an entity in the temporal knowledge graph.",
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
        "description": "Add a relationship between two named entities. Creates the entities if they don't exist. Time-stamped; can be ended/superseded later.",
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
        "description": "Query entities by name (substring, case-insensitive) and kind. Returns list with id, name, kind, aliases, notes.",
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
        "description": "Get all active relationships touching an entity at time `at` (default: now). Returns id, source_id, target_id, label, valid_from, valid_until, is_active.",
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
        "description": "Full history (active + ended) of relationships for an entity, newest first.",
        "inputSchema": {
            "type": "object",
            "properties": {"entity": {"type": "string"}},
            "required": ["entity"],
        },
    },

    # ----- Memory blocks (Layer 3) -----
    {
        "name": "brain_block_read",
        "description": "Read a memory block by name (persona, user, active_project, today_focus, open_questions, or your own).",
        "inputSchema": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        },
    },
    {
        "name": "brain_block_write",
        "description": "Replace a block's content (creates it if missing).",
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
        "description": "Append text to an existing block (preserves history).",
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
        "description": "List all memory blocks with name, char_count, updated_at.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "brain_seed_blocks",
        "description": "Seed the 5 default blocks (persona, user, active_project, today_focus, open_questions). Idempotent - skips blocks that already exist.",
        "inputSchema": {"type": "object", "properties": {}},
    },

    # ----- Injection scan (Layer 4) -----
    {
        "name": "brain_injection_scan",
        "description": "Run a one-shot injection scan on text. Returns is_clean, max_severity, pattern_hits, heuristic_hits, scan_id. Does NOT quarantine.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "brain_quarantine_list",
        "description": "List quarantined chunks. status: 'pending' (default), 'approved', 'rejected', or 'all'.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["pending", "approved", "rejected", "all"], "default": "pending"},
            },
        },
    },
    {
        "name": "brain_quarantine_review",
        "description": "Approve (false positive) or reject (true positive) a quarantined chunk by scan_id.",
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
        "description": "Layer 9: list chunks due for FSRS-6 spaced-repetition review (R(t,S) < 0.9). Public-domain math, no LLM call. Returns retrievability, stability, difficulty, urgency per chunk, sorted by urgency descending.",
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
        "description": "Layer 8: Ebbinghaus decay status (R = e^(-t/S)) for recent chunks, grouped by tier. Public-domain math (1885), no LLM call. Useful for hygiene review ('which knowledge is fading?').",
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
        "description": "Delete the top-k chunks matching a query. Use when you want to forget a topic, not just one chunk. Returns deleted_ids and what was matched before deletion.",
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
        "description": "Layer 13: exact substring match against the verbatim (pre-overlap) source text. Useful when you remember a phrase verbatim and want the chunk that contains it. Different from brain_recall (which is semantic+BM25).",
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
        "description": (
            "v0.11.0: Pull DREAMS.md + memory/dreaming/*.md into the brain as "
            "`semantic` tier. Idempotent. Returns new_entries, skipped, by_kind, sources."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "brain_dreaming_cycle",
        "description": (
            "v0.11.0: Distill high-importance episodic chunks into a new dream "
            "entry. Returns distilled_chunks, by_tier, output_files."
        ),
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
        "description": (
            "v0.11.0: Hermes /learn shim. Ingest + write to memory/learning/ + "
            "optionally invoke `hermes learn`. Returns chunk_id, written_to, "
            "hermes_invoked, hermes_output."
        ),
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
        "description": (
            "v0.11.0: OpenClaw Active Memory tool alias. Dispatches "
            "memory_query / memory_store / memory_recent / memory_forget. "
            "Returns {ok, tool, data, error}."
        ),
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


def handle(tool_name: str, args: dict) -> dict:
    """
    Dispatch a single MCP tool call to the Brain facade.

    Returns a JSON-serializable dict. Errors are caught and returned as
    {"error": "..."} so the MCP client can surface them cleanly.
    """
    try:
        brain = Brain()
        if tool_name == "brain_stats":
            return _serialize(brain.stats())
        if tool_name == "brain_remember":
            r = brain.remember(
                text=args["text"],
                source_path=args.get("source_path", "<openclaw>"),
                force_tier=args.get("force_tier"),
                skip_scan=args.get("skip_scan", False),
            )
            return _serialize(r)
        if tool_name == "brain_recall_verbatim":
            results = brain.recall_verbatim(
                query=args["query"],
                k=args.get("k", 5),
                tier=args.get("tier"),
                min_importance=args.get("min_importance"),
                rerank=args.get("rerank"),
                decay=args.get("decay"),
                tier_priors=args.get("tier_priors"),
                tier_priors_overrides=args.get("tier_priors_overrides"),
                fsrs=args.get("fsrs"),
            )
            return {"results": results}
        if tool_name == "brain_recall":
            results = brain.recall(
                query=args["query"],
                k=args.get("k", 5),
                tier=args.get("tier"),
                min_importance=args.get("min_importance"),
                rerank=args.get("rerank"),
                decay=args.get("decay"),
                tier_priors=args.get("tier_priors"),
                tier_priors_overrides=args.get("tier_priors_overrides"),
                fsrs=args.get("fsrs"),
            )
            return {"results": _serialize(results)}
        if tool_name == "brain_reflect":
            from src.memory import Memory
            import asyncio
            return asyncio.run(Memory().reflect(
                lookback_days=args.get("lookback_days", 7),
                max_chunks=args.get("max_chunks", 200),
            ))

        # Graph
        if tool_name == "brain_graph_entity":
            return brain.graph_upsert_entity(
                name=args["name"],
                kind=args.get("kind", "concept"),
                properties=args.get("properties"),
            )
        if tool_name == "brain_graph_relate":
            return brain.graph_add_relationship(
                source=args["source"],
                target=args["target"],
                label=args["label"],
                properties=args.get("properties"),
            )
        if tool_name == "brain_graph_query":
            return {"entities": brain.graph_query(name=args.get("name"), kind=args.get("kind"))}
        if tool_name == "brain_graph_relationships":
            return {"relationships": brain.graph_relationships(
                entity_name=args["entity"], at=args.get("at")
            )}
        if tool_name == "brain_graph_history":
            return {"history": brain.graph_history(entity_name=args["entity"])}

        # Blocks
        if tool_name == "brain_block_read":
            r = brain.block_read(args["name"])
            return r if r is not None else {"error": f"block not found: {args['name']}"}
        if tool_name == "brain_block_write":
            return brain.block_write(args["name"], args["text"])
        if tool_name == "brain_block_append":
            return brain.block_append(args["name"], args["text"])
        if tool_name == "brain_block_delete":
            return brain.block_delete(args["name"])
        if tool_name == "brain_block_list":
            return {"blocks": brain.block_list()}
        if tool_name == "brain_seed_blocks":
            return {"seeded": brain.seed_default_blocks()}

        # Quarantine
        if tool_name == "brain_injection_scan":
            return brain.injection_scan(args["text"])
        if tool_name == "brain_quarantine_list":
            return {"quarantined": brain.quarantine_list(status=args.get("status", "pending"))}
        if tool_name == "brain_quarantine_review":
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
            return brain.forget_by_query(
                query=args["query"],
                k=args.get("k", 5),
                tier=args.get("tier"),
            )
        if tool_name == "brain_search_verbatim":
            return {"matches": brain.search_verbatim(
                needle=args["needle"],
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
            return brain.learn(
                text=args["text"],
                force_tier=args.get("force_tier", "procedural"),
                source=args.get("source", "<hermes-/learn>"),
                metadata=args.get("metadata"),
                invoke_hermes=args.get("invoke_hermes", True),
            )
        if tool_name == "brain_active_memory":
            return brain.active_memory(
                tool=args["tool"],
                args=args.get("args", {}),
            )

        return {"error": f"unknown tool: {tool_name}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "tool": tool_name, "args": args}


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
            "description": "DuckBot brain: vector + graph + blocks + quarantine + dreaming + /learn + active-memory (39 tools)",
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
