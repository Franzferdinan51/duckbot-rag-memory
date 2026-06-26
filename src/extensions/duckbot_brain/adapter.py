"""Generic JSON-RPC MCP adapter for the DuckBot brain.

Speaks JSON-RPC over stdio so any MCP-compatible client can drive the
brain as a memory provider. Used by:
  - Claude Code (`~/.claude.json` mcp_servers entry)
  - Cursor (`~/.cursor/mcp.json`)
  - Codex CLI (`~/.codex/mcp.json`)
  - mcporter
  - Any other MCP client that takes a command + args + env config

This is NOT an OpenClaw native plugin — OpenClaw plugins run in-process
inside the Node gateway and can't load Python. OpenClaw users should
install the Node.js shim at `extensions/duckbot-memory/` instead, which
spawns the Python MCP server (`src/mcp_server.py`) as a subprocess.

The 12 tools exposed here delegate to the shared surface at
`src.extensions.tools`, so an agent author can rely on the same tool
names regardless of which platform they're on. The full 64-tool surface
is available via `python -m src.mcp_server` for admin / CLI use.

Pattern sources:
  - MCP spec (Model Context Protocol, Content-Length framed JSON-RPC)
  - OpenClaw active-memory extension (TypeScript, MIT) — for naming
    conventions, even though we run as a separate process.
"""

# MIT License — see LICENSE in the repository root.


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

# Shared core surface — same tool list as the Hermes plugin, the MCP
# server (subset), and any future adapter. See src/extensions/tools.py
# for the full contract.
from src.extensions import tools as _tools  # noqa: E402

logger = logging.getLogger("duckbot-brain-extension")


# -----------------------------------------------------------------------------
# Config (read once at startup from env or openclaw.plugin.json defaults)
# -----------------------------------------------------------------------------


_DEFAULT_K = int(os.environ.get("DUCKBOT_BRAIN_DEFAULT_K", "5"))
_ENABLE_RERANK = os.environ.get("DUCKBOT_BRAIN_RERANK", "0") == "1"
_ENABLE_DECAY = os.environ.get("DUCKBOT_BRAIN_DECAY", "0") == "1"


# -----------------------------------------------------------------------------
# Back-compat shim: tests + older callers reference _BRAIN directly
# (`adapter._BRAIN = fake_brain` then expect _call_tool to use it).
# The new tools module owns the canonical singleton; this attribute
# is a per-adapter override that wins over the shared singleton.
# -----------------------------------------------------------------------------


_BRAIN: Optional[Any] = None  # Any = Brain; lazy-imported to avoid cycle


def _get_brain():  # type: ignore[no-untyped-def]
    """Per-adapter override first, then shared singleton."""
    if _BRAIN is not None:
        return _BRAIN
    return _tools._get_brain()


def _resolve_brain_override(brain: Any) -> None:
    """Test helper: pass an explicit brain (or None to clear)."""
    global _BRAIN
    _BRAIN = brain


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
    """MCP-style tool list response.

    Delegates to the shared core surface in `src.extensions.tools` so
    the OpenClaw extension, Hermes plugin, and any future adapter stay
    in lock-step. v0.14.0 (was 8, now 9: added `brain_wake_up` so the
    skill files' "call brain_wake_up at session start" instruction
    actually works).
    """
    tools = _tools.tool_schemas()
    # Apply the per-process default-rerank/decay overrides from env
    # (configurable via openclaw.plugin.json `enableRerank` / `enableDecay`).
    for t in tools:
        if t["name"] == "brain_recall":
            schema = t["inputSchema"]
            if "properties" in schema and "k" in schema["properties"]:
                schema["properties"]["k"]["default"] = _DEFAULT_K
            if "properties" in schema and "rerank" in schema["properties"]:
                schema["properties"]["rerank"]["default"] = _ENABLE_RERANK
            if "properties" in schema and "decay" in schema["properties"]:
                schema["properties"]["decay"]["default"] = _ENABLE_DECAY
    return {"tools": tools}


def _call_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Execute one tool call and return MCP-style content blocks.

    v0.14.0: delegates to the shared dispatch in `src.extensions.tools`.
    Per-tool rate limit is enforced before dispatch (matches the MCP
    server's behavior so a misbehaving OpenClaw agent can't bypass the
    limit by retrying). DUCKBOT_RATELIMIT_DISABLE=1 turns this off.
    """
    # Per-tool rate limit (same shape as mcp_server._check_rate_limit_or_error).
    rl_err = _tools.check_rate_limit(name)
    if rl_err is not None:
        return _content(rl_err, is_error=True)

    brain = _get_brain()
    # If a per-adapter _BRAIN override is set (test fixture), thread it
    # into dispatch via a thread-local. Simplest: monkeypatch the
    # shared _tools._get_brain temporarily.
    if _BRAIN is not None:
        original = _tools._BRAIN
        _tools._BRAIN = _BRAIN
        try:
            result = _tools.dispatch(name, args)
        finally:
            _tools._BRAIN = original
    else:
        result = _tools.dispatch(name, args)
    is_error = bool(isinstance(result, dict) and result.get("error"))
    return _content(result, is_error=is_error)


def _content(payload: Any, is_error: bool = False) -> Dict[str, Any]:
    """MCP-style content blocks. v0.14.0: accepts is_error flag."""
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(payload, default=str, indent=2),
            }
        ],
        "isError": is_error,
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
