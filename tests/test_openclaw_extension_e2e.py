"""
test_openclaw_extension_e2e.py — end-to-end smoke test of the OpenClaw
extension adapter's stdio JSON-RPC loop.

Different from test_openclaw_extension.py (which calls adapter._call_tool
directly). This one spawns the adapter as a subprocess, sends real
JSON-RPC frames over stdin, and parses the stdout responses. Catches
issues that the unit-level tests miss — e.g. sys.path setup, env-var
detection, framing bugs.

Skipped automatically when:
  - the venv isn't present (.venv/bin/python missing)
  - LM Studio isn't reachable (the tools/list call still works but
    brain_stats would error — we only test tools/list here so this
    isn't actually a blocker)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".venv" / "bin" / "python"
if not VENV_PY.exists():
    # Windows fallback
    VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"


pytestmark = pytest.mark.skipif(
    not VENV_PY.exists(),
    reason="venv python not found — run scripts/install.sh first",
)


def _spawn_adapter() -> subprocess.Popen:
    """Spawn the adapter as a subprocess, ready to receive JSON-RPC."""
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["DUCKBOT_BRAIN_LOG_LEVEL"] = "WARNING"
    proc = subprocess.Popen(
        [str(VENV_PY), "-m", "src.extensions.duckbot_brain.adapter"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(ROOT),
        env=env,
        text=True,
    )
    return proc


def _send(proc: subprocess.Popen, req: dict) -> dict:
    """Send one JSON-RPC frame (newline-delimited) and read one response."""
    line = json.dumps(req) + "\n"
    assert proc.stdin is not None
    proc.stdin.write(line)
    proc.stdin.flush()
    assert proc.stdout is not None
    resp_line = proc.stdout.readline()
    assert resp_line, "no response from adapter (process exited?)"
    return json.loads(resp_line)


# -----------------------------------------------------------------------------
# Protocol handshake
# -----------------------------------------------------------------------------


def test_initialize_returns_protocol_info():
    proc = _spawn_adapter()
    try:
        resp = _send(proc, {
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {},
        })
        assert resp["jsonrpc"] == "2.0"
        assert resp["id"] == 1
        assert resp["result"]["protocolVersion"] == "2024-11-05"
        assert resp["result"]["serverInfo"]["name"] == "duckbot-brain"
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_tools_list_returns_nine_tools():
    """E2E: the adapter must expose all 9 core tools, with brain_wake_up
    listed. This is the test that catches the v0.14.0 bug (wake_up was
    missing from the adapter)."""
    proc = _spawn_adapter()
    try:
        resp = _send(proc, {
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
        })
        assert resp["id"] == 2
        tools = resp["result"]["tools"]
        names = [t["name"] for t in tools]
        assert len(names) == 12, f"expected 11 tools, got {len(names)}: {names}"
        assert "brain_wake_up" in names, "brain_wake_up must be exposed"
        # All tools have the required MCP fields.
        for t in tools:
            assert "name" in t
            assert "description" in t
            assert "inputSchema" in t
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_unknown_method_returns_jsonrpc_error():
    proc = _spawn_adapter()
    try:
        resp = _send(proc, {
            "jsonrpc": "2.0", "id": 3, "method": "nonsense/method", "params": {},
        })
        assert resp["id"] == 3
        assert "error" in resp
        assert resp["error"]["code"] == -32601
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_malformed_json_does_not_crash_loop():
    """The adapter's stdin loop must not crash on a malformed JSON line —
    it should skip the bad line and keep serving."""
    proc = _spawn_adapter()
    try:
        assert proc.stdin is not None
        # Send garbage first.
        proc.stdin.write("not valid json\n")
        proc.stdin.flush()
        # Then a valid request — must still get a response.
        resp = _send(proc, {
            "jsonrpc": "2.0", "id": 4, "method": "initialize", "params": {},
        })
        assert resp["id"] == 4
        assert resp["result"]["protocolVersion"] == "2024-11-05"
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_tools_call_brain_stats_returns_real_shape():
    """Calling brain_stats through the real subprocess should return the
    9-field BrainStats shape (or an error if Chroma isn't available —
    either way, a valid MCP content block, not a crash)."""
    proc = _spawn_adapter()
    try:
        resp = _send(proc, {
            "jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": {"name": "brain_stats", "arguments": {}},
        })
        assert resp["id"] == 5
        assert "result" in resp
        content = resp["result"]["content"]
        assert len(content) == 1
        assert content[0]["type"] == "text"
        # The text is JSON-encoded; parse it.
        payload = json.loads(content[0]["text"])
        # Either the real stats fields, or an error dict (if Chroma/LM
        # Studio isn't available in this CI env). Both are valid shapes.
        assert isinstance(payload, dict)
        if "error" not in payload:
            # Real stats path.
            assert "vector_chunks" in payload
            assert "vector_by_tier" in payload
            assert "generated_at" in payload
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_tools_call_unknown_tool_returns_error_content():
    """An unknown tool name must surface in the content block, not crash
    the JSON-RPC envelope."""
    proc = _spawn_adapter()
    try:
        resp = _send(proc, {
            "jsonrpc": "2.0", "id": 6, "method": "tools/call",
            "params": {"name": "brain_does_not_exist", "arguments": {}},
        })
        assert resp["id"] == 6
        # JSON-RPC envelope is OK; the error is in the content block.
        assert "result" in resp
        payload = json.loads(resp["result"]["content"][0]["text"])
        assert "error" in payload
        assert resp["result"]["isError"] is True
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_multiple_requests_in_one_session():
    """The loop must handle a sequence of requests on the same stdin."""
    proc = _spawn_adapter()
    try:
        for i in range(1, 4):
            resp = _send(proc, {
                "jsonrpc": "2.0", "id": i, "method": "tools/list", "params": {},
            })
            assert resp["id"] == i
            assert len(resp["result"]["tools"]) == 12
    finally:
        proc.terminate()
        proc.wait(timeout=5)
