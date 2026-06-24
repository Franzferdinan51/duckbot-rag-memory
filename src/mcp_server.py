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
]  


# -----------------------------------------------------------------------------
# Connector tools (Layers 1-4 + dashboard) - backed by the framework-agnostic
# Brain facade. OpenClaw picks them up automatically because they're in TOOLS.
# -----------------------------------------------------------------------------

def _import_connector_tools() -> tuple[list[dict], dict]:
    """Import the OpenClaw connector's TOOL_DEFINITIONS and dispatchers.
    Returns (extra_tools, extra_handlers)."""
    from src.connectors.openclaw import TOOL_DEFINITIONS, handle as _handle
    extra_tools = list(TOOL_DEFINITIONS)
    extra_handlers = {t["name"]: (lambda args, h=_handle: h(t["name"], args)) for t in TOOL_DEFINITIONS}
    return extra_tools, extra_handlers


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
}

# Register the 18 connector tools (graph + blocks + quarantine + scan).
# Called after HANDLERS is defined so the dispatch table is ready.
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
                        "serverInfo": {"name": "duckbot-memory", "version": "0.11.1"},
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
                        result = asyncio.run(handler(args))
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


def main():
    p = argparse.ArgumentParser(description="DuckBot memory MCP server")
    p.add_argument("--http", type=int, help="run as HTTP server on PORT (instead of stdio)")
    args = p.parse_args()
    if args.http:
        # Minimal HTTP wrapper using aiohttp or similar
        # For now just print a hint
        print(f"HTTP mode not yet implemented. Use stdio for now. (Would listen on {args.http})", file=sys.stderr)
        sys.exit(1)
    mcp_stdio()


if __name__ == "__main__":
    main()
