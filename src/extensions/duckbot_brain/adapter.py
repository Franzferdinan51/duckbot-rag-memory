"""OpenClaw extension adapter for the DuckBot brain.

Speaks JSON-RPC over stdio so OpenClaw (or any MCP-compatible client)
can drive the brain as a memory provider.

This is the Python sibling of the Hermes plugin at
src/plugins/memory/duckbot_brain/__init__.py — both implement the same
Brain API; only the protocol differs.

Pattern sources:
  - OpenClaw active-memory extension (extensions/active-memory/index.ts)
    https://github.com/openclaw/openclaw/blob/main/extensions/active-memory/
  - mcporter's stdio JSON-RPC adapter pattern
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

# Make `src.*` importable when this module is invoked as
# `python -m src.extensions.duckbot_brain.adapter` from the repo root.
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[3]  # src/extensions/duckbot_brain/adapter.py → repo
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.connectors.base import Brain  # noqa: E402

logger = logging.getLogger("duckbot-brain-extension")


# -----------------------------------------------------------------------------
# Config (read once at startup from env or openclaw.plugin.json defaults)
# -----------------------------------------------------------------------------


_DEFAULT_K = int(os.environ.get("DUCKBOT_BRAIN_DEFAULT_K", "5"))
_ENABLE_RERANK = os.environ.get("DUCKBOT_BRAIN_RERANK", "0") == "1"
_ENABLE_DECAY = os.environ.get("DUCKBOT_BRAIN_DECAY", "0") == "1"


# -----------------------------------------------------------------------------
# Brain singleton
# -----------------------------------------------------------------------------


_BRAIN: Optional[Brain] = None


def _get_brain() -> Brain:
    global _BRAIN
    if _BRAIN is None:
        _BRAIN = Brain()
    return _BRAIN


# -----------------------------------------------------------------------------
# JSON-RPC handler
# -----------------------------------------------------------------------------


def handle_request(req: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch a JSON-RPC 2.0 request and return the response dict."""
    method = req.get("method", "")
    params = req.get("params", {}) or {}
    req_id = req.get("id")

    try:
        if method == "tools/list":
            return _ok(req_id, _tool_schemas())
        if method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments") or {}
            return _ok(req_id, _call_tool(tool_name, tool_args))
        if method == "initialize":
            return _ok(req_id, {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "duckbot-brain", "version": "0.1.0"},
                "capabilities": {"tools": {}},
            })
        return _err(req_id, -32601, f"method not found: {method}")
    except Exception as e:
        logger.warning("handler error in %s: %s", method, e)
        logger.debug(traceback.format_exc())
        return _err(req_id, -32603, f"internal error: {e}")


def _ok(req_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _err(req_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


# -----------------------------------------------------------------------------
# Tool surface
# -----------------------------------------------------------------------------


def _tool_schemas() -> Dict[str, Any]:
    """MCP-style tool list response."""
    return {
        "tools": [
            {
                "name": "brain_recall",
                "description": "Hybrid retrieval (vector + BM25 + RRF). Optionally rerank=true for cross-encoder boost, decay=true for Ebbinghaus retention weighting.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": _DEFAULT_K},
                        "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"]},
                        "min_importance": {"type": "number"},
                        "rerank": {"type": "boolean", "default": _ENABLE_RERANK},
                        "decay": {"type": "boolean", "default": _ENABLE_DECAY},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "brain_recall_verbatim",
                "description": "Returns the original (pre-overlap, pre-prefix) source text.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": _DEFAULT_K},
                        "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"]},
                        "rerank": {"type": "boolean", "default": False},
                        "decay": {"type": "boolean", "default": False},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "brain_remember",
                "description": "Persist text to the brain. Non-blocking; the watcher will pick it up if needed.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "source": {"type": "string"},
                        "tier": {"type": "string", "enum": ["working", "episodic", "semantic", "procedural"]},
                    },
                    "required": ["text"],
                },
            },
            {
                "name": "brain_stats",
                "description": "Return brain stats.",
                "inputSchema": {"type": "object", "properties": {}},
            },
        ]
    }


def _call_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one tool call and return MCP-style content blocks."""
    brain = _get_brain()
    if name == "brain_recall":
        results = brain.recall(
            query=args["query"],
            k=args.get("k", _DEFAULT_K),
            tier=args.get("tier"),
            min_importance=args.get("min_importance"),
            rerank=args.get("rerank") or False,
            decay=args.get("decay") or False,
        )
        return _content(_serialize_recall(results))
    if name == "brain_recall_verbatim":
        results = brain.recall_verbatim(
            query=args["query"],
            k=args.get("k", _DEFAULT_K),
            tier=args.get("tier"),
            rerank=args.get("rerank") or False,
            decay=args.get("decay") or False,
        )
        return _content(results)
    if name == "brain_remember":
        # Non-blocking — return immediately, the actual ingest runs in the
        # background. The caller does not need to wait.
        import asyncio
        text = args["text"]
        source = args.get("source") or "openclaw-extension://ad-hoc"
        try:
            # Spin off the remember; we do not await.
            import threading
            def _do():
                try:
                    asyncio.run(brain.remember(text=text, source_path=source))
                except Exception as e:
                    logger.warning("background remember failed: %s", e)
            threading.Thread(target=_do, daemon=True).start()
            return _content({"status": "queued", "source": source})
        except Exception as e:
            return _content({"error": str(e)})
    if name == "brain_stats":
        s = brain.stats()
        return _content({
            "chunks_per_tier": s.chunks_per_tier if hasattr(s, "chunks_per_tier") else {},
            "last_query_at": s.last_query_at if hasattr(s, "last_query_at") else None,
        })
    return _content({"error": f"unknown tool: {name}"})


def _serialize_recall(results) -> list:
    return [
        {
            "chunk_id": r.chunk_id,
            "text": r.text,
            "tier": r.tier,
            "importance": r.importance,
            "score": r.score,
            "source_path": r.source_path,
            "metadata": r.metadata,
        }
        for r in results
    ]


def _content(payload: Any) -> Dict[str, Any]:
    """MCP-style content blocks."""
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, default=str, indent=2),
            }
        ],
        "isError": False,
    }


# -----------------------------------------------------------------------------
# Stdio JSON-RPC loop
# -----------------------------------------------------------------------------


def main() -> int:
    """Read JSON-RPC requests from stdin, write responses to stdout.

    One request per line (newline-delimited JSON). Handles the
    Content-Length header style (used by MCP stdio) too.
    """
    logging.basicConfig(
        level=os.environ.get("DUCKBOT_BRAIN_LOG_LEVEL", "WARNING"),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    logger.info("duckbot-brain extension starting (Python %s)", sys.version.split()[0])

    # MCP stdio framing: messages are preceded by "Content-Length: N\r\n\r\n".
    # We accept BOTH newline-delimited and Content-Length styles for
    # compatibility with the various OpenClaw transports.
    while True:
        try:
            line = sys.stdin.readline()
        except (EOFError, KeyboardInterrupt):
            return 0
        if not line:
            return 0
        line = line.strip()
        if not line:
            continue
        # If it's a Content-Length header, read the body.
        if line.lower().startswith("content-length:"):
            try:
                n = int(line.split(":", 1)[1].strip())
                sys.stdin.readline()  # consume blank line
                body = sys.stdin.read(n)
                req = json.loads(body)
            except Exception as e:
                logger.warning("malformed framed request: %s", e)
                continue
        else:
            try:
                req = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning("malformed JSON: %s", e)
                continue

        resp = handle_request(req)
        out = json.dumps(resp)
        sys.stdout.write(out + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    sys.exit(main())
