"""
connectors/openclaw.py - OpenClaw integration for the DuckBot brain.

.. deprecated:: v0.14.0
    Prefer `src.extensions.duckbot_brain.adapter` (the stdio JSON-RPC
    OpenClaw extension, 9-tool core agent surface) or
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
            # Validate required args — previously raised KeyError on empty/missing.
            # (The canonical `remember` handler in src/mcp_server.py has richer
            # validation + skill_candidate support; this is a back-compat alias.)
            if not args.get("text") or not str(args["text"]).strip():
                return {"error": "text must be a non-empty string", "tool": tool_name}
            r = brain.remember(
                text=args["text"],
                source_path=args.get("source_path", "<openclaw>"),
                force_tier=args.get("force_tier"),
                skip_scan=args.get("skip_scan", False),
            )
            return _serialize(r)
        if tool_name == "brain_recall_verbatim":
            if not args.get("query"):
                return {"error": "query is required", "tool": tool_name}
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
            # Convert VerbatimResult dataclasses to dicts so the MCP
            # server's json.dumps can serialize the response. (The native
            # mcp_server.py recall_verbatim handler has the same issue.)
            return {"results": [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in results]}
        if tool_name == "brain_recall":
            if not args.get("query"):
                return {"error": "query is required", "tool": tool_name}
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
            if not args.get("query"):
                return {"error": "query is required", "tool": tool_name}
            return brain.forget_by_query(
                query=args["query"],
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
