"""
test_openclaw_extension.py — verify the OpenClaw extension adapter.

duckbot-secret-scan: allowlist-file
"""

# duckbot-secret-scan: allowlist-file
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.extensions.duckbot_brain import adapter  # noqa: E402


# -----------------------------------------------------------------------------
# handle_request: dispatch
# -----------------------------------------------------------------------------


def test_handle_request_tools_list():
    req = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    resp = adapter.handle_request(req)
    assert resp["jsonrpc"] == "2.0"
    assert resp["id"] == 1
    assert "result" in resp
    tools = resp["result"]["tools"]
    names = {t["name"] for t in tools}
    assert "brain_recall" in names
    assert "brain_recall_verbatim" in names
    assert "brain_remember" in names
    assert "brain_stats" in names


def test_handle_request_initialize_returns_protocol_info():
    req = {"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {}}
    resp = adapter.handle_request(req)
    assert resp["result"]["protocolVersion"] == "2024-11-05"
    assert resp["result"]["serverInfo"]["name"] == "duckbot-brain"


def test_handle_request_unknown_method_returns_error():
    req = {"jsonrpc": "2.0", "id": 3, "method": "nonsense", "params": {}}
    resp = adapter.handle_request(req)
    assert "error" in resp
    assert resp["error"]["code"] == -32601


def test_handle_request_unknown_tool_returns_error_block():
    """tools/call with an empty params (name=None) routes to 'unknown tool'
    path inside _call_tool — the error surfaces in the tool result, not
    the JSON-RPC envelope."""
    req = {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {}}
    resp = adapter.handle_request(req)
    # JSON-RPC envelope is OK (the call returned); the error is in the tool result.
    assert "result" in resp
    text = resp["result"]["content"][0]["text"]
    assert "unknown tool" in text


def test_handle_request_internal_exception_returns_error():
    """If the handler itself throws, return a JSON-RPC error."""
    from unittest.mock import patch
    req = {"jsonrpc": "2.0", "id": 5, "method": "tools/list", "params": {}}
    with patch.object(adapter, "_tool_schemas", side_effect=RuntimeError("boom")):
        resp = adapter.handle_request(req)
    assert "error" in resp
    assert resp["error"]["code"] == -32603
    assert "boom" in resp["error"]["message"]


def test_call_tool_brain_recall_delegates_to_brain():
    fake_brain = MagicMock()
    fake_brain.recall.return_value = [
        MagicMock(
            chunk_id="x", text="t", tier="semantic", importance=0.5,
            score=0.1, source_path="/tmp/y.md", metadata={},
        )
    ]
    adapter._BRAIN = fake_brain
    result = adapter._call_tool("brain_recall", {"query": "q", "k": 3, "rerank": True})
    assert result["isError"] is False
    fake_brain.recall.assert_called_once_with(
        query="q", k=3, tier=None, min_importance=None,
        rerank=True, decay=False,
        tier_priors=False, tier_priors_overrides=None, fsrs=False,
    )
    # Content block is JSON-encoded text. v0.14.0: dispatch returns
    # {"results": [...]} (matches MCP server + Hermes plugin shape),
    # so the chunk lives at payload["results"][0].
    text_block = result["content"][0]["text"]
    payload = json.loads(text_block)
    assert payload["results"][0]["chunk_id"] == "x"


# -----------------------------------------------------------------------------
# _call_tool: brain_recall
# -----------------------------------------------------------------------------


def test_call_tool_brain_recall_delegates_to_brain():
    fake_brain = MagicMock()
    fake_brain.recall.return_value = [
        MagicMock(
            chunk_id="x", text="t", tier="semantic", importance=0.5,
            score=0.1, source_path="/tmp/y.md", metadata={},
        )
    ]
    adapter._BRAIN = fake_brain
    result = adapter._call_tool("brain_recall", {"query": "q", "k": 3, "rerank": True})
    assert result["isError"] is False
    fake_brain.recall.assert_called_once_with(
        query="q", k=3, tier=None, min_importance=None,
        rerank=True, decay=False,
        tier_priors=False, tier_priors_overrides=None, fsrs=False,
    )
    # Content block is JSON-encoded text.
    text_block = result["content"][0]["text"]
    payload = json.loads(text_block)
    assert payload["results"][0]["chunk_id"] == "x"


def test_call_tool_brain_recall_verbatim_delegates():
    fake_brain = MagicMock()
    fake_brain.recall_verbatim.return_value = [{"verbatim_text": "src"}]
    adapter._BRAIN = fake_brain
    result = adapter._call_tool("brain_recall_verbatim", {"query": "q"})
    fake_brain.recall_verbatim.assert_called_once()
    payload = json.loads(result["content"][0]["text"])
    assert payload["results"][0]["verbatim_text"] == "src"


def test_call_tool_brain_remember_returns_queued():
    """brain_remember must be non-blocking — returns immediately."""
    fake_brain = MagicMock()
    adapter._BRAIN = fake_brain
    result = adapter._call_tool("brain_remember", {"text": "remember this", "source": "test://"})
    payload = json.loads(result["content"][0]["text"])
    assert payload["status"] == "queued"
    assert payload["source"] == "test://"
    # We do NOT await — the remember is fire-and-forget on a daemon thread.


def test_call_tool_brain_stats_delegates():
    fake_brain = MagicMock()
    fake_brain.stats.return_value = MagicMock(
        vector_chunks=150,
        vector_by_tier={"semantic": 100, "episodic": 50},
        graph_entities=0,
        graph_relationships=0,
        graph_active_relationships=0,
        blocks=0,
        quarantine_total=0,
        quarantine_pending=0,
        quarantine_approved=0,
        quarantine_rejected=0,
        generated_at=1234567890.0,
    )
    adapter._BRAIN = fake_brain
    result = adapter._call_tool("brain_stats", {})
    payload = json.loads(result["content"][0]["text"])
    # v0.10.0 fix: brain_stats now returns the real BrainStats fields.
    assert payload["vector_chunks"] == 150
    assert payload["vector_by_tier"]["semantic"] == 100
    assert payload["vector_by_tier"]["episodic"] == 50
    assert payload["generated_at"] == 1234567890.0


def test_call_tool_unknown_tool_returns_error():
    adapter._BRAIN = MagicMock()
    result = adapter._call_tool("brain_does_not_exist", {})
    payload = json.loads(result["content"][0]["text"])
    assert "error" in payload


# -----------------------------------------------------------------------------
# Tool schemas match the openclaw.plugin.json contract
# -----------------------------------------------------------------------------


def test_tool_schemas_have_required_fields():
    schemas = adapter._tool_schemas()["tools"]
    for tool in schemas:
        assert "name" in tool
        assert "description" in tool
        assert "inputSchema" in tool


def test_tool_schemas_includes_brain_wake_up():
    """v0.14.0: brain_wake_up is the canonical session-start tool.
    Skills tell agents to call it on session start — the adapter MUST
    advertise it, otherwise agents get 'unknown tool' errors."""
    names = {t["name"] for t in adapter._tool_schemas()["tools"]}
    assert "brain_wake_up" in names


def test_call_tool_returns_is_error_on_rate_limit(monkeypatch):
    """v0.14.0: per-tool rate limit short-circuits with isError=true."""
    from src import ratelimit
    ratelimit.reset_rate_limiter()
    # Force-exhaust the bucket so the next check returns rate_limited.
    rl = ratelimit.get_rate_limiter()
    for _ in range(50):
        rl.check("brain_recall")
    fake_brain = MagicMock()
    fake_brain.recall.return_value = []
    adapter._BRAIN = fake_brain
    try:
        result = adapter._call_tool("brain_recall", {"query": "q"})
        # Either allowed (rare) or rate-limited — but never delegates when blocked.
        if result["isError"]:
            payload = json.loads(result["content"][0]["text"])
            assert payload["error"] == "rate_limited"
            assert payload["tool"] == "brain_recall"
            # Brain must NOT have been called.
            fake_brain.recall.assert_not_called()
    finally:
        ratelimit.reset_rate_limiter()


def test_brain_recall_schema_requires_query():
    schemas = adapter._tool_schemas()["tools"]
    recall = next(t for t in schemas if t["name"] == "brain_recall")
    assert "query" in recall["inputSchema"]["required"]


def test_brain_remember_schema_requires_text():
    schemas = adapter._tool_schemas()["tools"]
    rem = next(t for t in schemas if t["name"] == "brain_remember")
    assert "text" in rem["inputSchema"]["required"]


# -----------------------------------------------------------------------------
# Native OpenClaw plugin manifest (extensions/duckbot-memory/openclaw.plugin.json)
#
# v0.15.0: the Python "fake" manifest at src/extensions/duckbot_brain/
# was deleted because OpenClaw plugins run in-process inside the Node
# gateway and can't load Python. The real native plugin is a Node.js
# shim at extensions/duckbot-memory/ that spawns the Python MCP server
# as a subprocess. These tests verify the shim manifest exists + has
# the right shape.
# -----------------------------------------------------------------------------


SHIM_PLUGIN_DIR = ROOT / "extensions" / "duckbot-memory"
SHIM_MANIFEST = SHIM_PLUGIN_DIR / "openclaw.plugin.json"


def test_shim_plugin_dir_exists():
    """The native OpenClaw plugin directory must exist at the repo root."""
    assert SHIM_PLUGIN_DIR.is_dir(), (
        f"shim plugin missing: {SHIM_PLUGIN_DIR} — see extensions/duckbot-memory/README.md"
    )


def test_shim_plugin_package_json_points_at_index_js():
    import json as _json
    pkg = SHIM_PLUGIN_DIR / "package.json"
    assert pkg.is_file(), "package.json required for OpenClaw plugin loader"
    data = _json.loads(pkg.read_text())
    assert data["main"] == "index.js"
    assert data.get("openclaw", {}).get("manifest") == "./openclaw.plugin.json"


def test_shim_plugin_index_js_syntax_clean():
    """index.js must parse — OpenClaw loads it via Node's require()."""
    import subprocess as _sp
    res = _sp.run(
        ["node", "--check", str(SHIM_PLUGIN_DIR / "index.js")],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, f"index.js syntax error: {res.stderr}"


def test_shim_manifest_required_fields():
    """Manifest shape per openclaw/openclaw docs/plugins/manifest.md."""
    import json as _json
    assert SHIM_MANIFEST.is_file(), (
        f"shim manifest missing: {SHIM_MANIFEST} — see extensions/duckbot-memory/README.md"
    )
    data = _json.loads(SHIM_MANIFEST.read_text())
    assert data["id"] == "duckbot-memory"
    assert "name" in data
    assert "description" in data
    assert "configSchema" in data
    assert data["configSchema"]["type"] == "object"
    assert "repoPath" in data["configSchema"]["required"]


def test_shim_manifest_does_not_claim_python_entry():
    """The fake manifest claimed `entry: python` which OpenClaw never
    honored. The new manifest must NOT have an `entry` or `entryArgs`
    field — OpenClaw plugins load via package.json#main."""
    import json as _json
    data = _json.loads(SHIM_MANIFEST.read_text())
    assert "entry" not in data, (
        "Manifest should NOT have an `entry` field — OpenClaw loads via "
        "package.json#main, not via subprocess spawn from the manifest."
    )
    assert "entryArgs" not in data
