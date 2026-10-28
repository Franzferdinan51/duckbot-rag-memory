"""Bridge mode (v0.16.0): the duckbot-brain provider must stay live in
interpreters that cannot import chromadb (the Hermes gateway's isolated
bundled Python) by forwarding Brain calls to src.brain_worker running
under the repo venv Python.

Two layers:
- Routing tests use a fake worker client — no chromadb needed, they run
  anywhere the plugin module imports.
- The worker protocol test spawns the real worker against a TEMP data
  dir (DUCKBOT_BRAIN_DATA_DIR) and exercises only store-level ops
  (ping / stats / dispatch-stats) that need no embedding provider, so
  it runs offline. It skips when chromadb isn't importable here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plugins.memory.duckbot_brain import (  # noqa: E402
    DuckBotBrainProvider,
    _BridgeBrain,
)


class FakeClient:
    """_WorkerClient stand-in that records calls and returns canned data."""

    def __init__(self) -> None:
        self.calls: list = []
        self.closed = False

    def call(self, op, args=None, timeout=None):
        self.calls.append((op, args))
        if op == "ping":
            return {"pong": True, "chromadb": "fake"}
        if op == "dispatch":
            return {"echo": args}
        if op == "recall":
            return [{
                "chunk_id": "c1", "text": "hello memory", "tier": "semantic",
                "importance": 0.5, "score": 0.9, "source_path": "x.md",
                "metadata": {"source_path": "x.md"},
            }]
        if op == "remember":
            return {"stored": True, "chunk_id": "c2", "tier": "episodic"}
        if op == "wake_up":
            return {"memories": [], "stats": {}}
        if op == "stats":
            return {"vector_chunks": 42}
        raise ValueError(op)

    def close(self) -> None:
        self.closed = True


def _bridge_provider() -> DuckBotBrainProvider:
    p = DuckBotBrainProvider()
    p._avail_probed = True
    p._avail_result = True
    p._bridge_mode = True
    p._worker = FakeClient()
    return p


def test_handle_tool_call_routes_to_worker_dispatch():
    p = _bridge_provider()
    out = json.loads(p.handle_tool_call("brain_stats", {"verbose": True}))
    assert out == {"echo": {"name": "brain_stats", "args": {"verbose": True}}}
    assert p._worker.calls[-1][0] == "dispatch"


def test_bridge_brain_recall_namespaces():
    p = _bridge_provider()
    brain = p._get_brain()
    assert isinstance(brain, _BridgeBrain)
    rows = brain.recall("anything", k=3, rerank=False, decay=False)
    assert rows[0].text == "hello memory"
    assert rows[0].tier == "semantic"
    assert rows[0].metadata == {"source_path": "x.md"}


def test_bridge_brain_remember_and_wake_up():
    p = _bridge_provider()
    brain = p._get_brain()
    res = brain.remember(text="note", source_path="test://x")
    assert res.stored is True
    assert brain.wake_up(k=8) == {"memories": [], "stats": {}}
    assert brain.stats().vector_chunks == 42


def test_shutdown_closes_worker():
    p = _bridge_provider()
    worker = p._worker
    p.shutdown()
    assert worker.closed is True


def test_venv_python_missing_returns_none(tmp_path):
    from src.plugins.memory.duckbot_brain import _venv_python
    assert _venv_python(str(tmp_path)) is None


@pytest.mark.skipif(
    pytest.importorskip("chromadb", reason="worker needs chromadb") is None,
    reason="chromadb unavailable",
)
def test_worker_protocol_temp_store(tmp_path):
    env = dict(**{k: v for k, v in __import__("os").environ.items()
                  if k != "PYTHONPATH"})
    env["DUCKBOT_BRAIN_DATA_DIR"] = str(tmp_path / "data")
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, "-m", "src.brain_worker"],
        cwd=str(ROOT), env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1,
    )
    seq = 0

    def call(op, args=None):
        nonlocal seq
        seq += 1
        proc.stdin.write(json.dumps(
            {"id": seq, "op": op, "args": args or {}}) + "\n")
        proc.stdin.flush()
        resp = json.loads(proc.stdout.readline())
        assert resp["id"] == seq
        return resp

    try:
        r = call("ping")
        assert r["ok"] and r["result"]["pong"] is True
        r = call("stats")
        assert r["ok"] and "vector_chunks" in r["result"]
        r = call("dispatch", {"name": "brain_stats", "args": {}})
        assert r["ok"] and isinstance(r["result"], dict)
        r = call("nope")
        assert r["ok"] is False and "unknown op" in r["error"]
    finally:
        try:
            call("shutdown")
        except Exception:
            pass
        proc.wait(timeout=10)
