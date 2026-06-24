"""
test_v0_11_integration.py — tests for the v0.11.0 integrations:
  - Track 2: Hermes /learn shim
  - Track 3: OpenClaw dreaming bridge
  - Track 4: Active Memory tool aliases
  - Track 5: Hermes hooks (on_pre_compress, on_memory_write)

All tests use a tempdir for dreaming state and an in-memory Memory
backend where possible. We don't shell out to `hermes` in tests.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def tmp_dreaming(tmp_path):
    """A complete OpenClaw dreaming surface in a temp dir."""
    dream_dir = tmp_path / "memory" / "dreaming" / "deep"
    dream_dir.mkdir(parents=True)
    diary = tmp_path / "DREAMS.md"
    state = tmp_path / "memory" / "dreaming" / ".brain_state.json"
    return {
        "root": tmp_path,
        "diary": diary,
        "dreaming_dir": tmp_path / "memory" / "dreaming",
        "state": state,
    }


@pytest.fixture
def tmp_learning_dir(tmp_path):
    p = tmp_path / "memory" / "learning"
    p.mkdir(parents=True)
    return p


# -----------------------------------------------------------------------------
# Track 2 — Hermes /learn shim
# -----------------------------------------------------------------------------


def _fake_remember_result(chunk_id="test-chunk-id", importance=0.8, tier=None):
    """Build a RememberResult matching the real dataclass signature."""
    from src.memory import RememberResult, Tier
    return RememberResult(
        text="fake text",
        chunk_id=chunk_id,
        tier=tier or Tier.PROCEDURAL,
        confidence=1.0,
        importance=importance,
        stored=True,
    )


def test_learn_bridge_writes_to_learning_dir(tmp_path, tmp_learning_dir):
    """Hermes /learn shim should write to memory/learning/<date>.md."""
    from src.connectors.learn import LearnBridge
    from src.memory import Memory

    # We can't run the real Memory here (needs LM Studio), so we test the
    # file-writing side directly by patching the memory call.
    async def fake_remember(self, *a, **kw):
        return _fake_remember_result(chunk_id="test-chunk-id")

    async def run():
        mem = Memory.__new__(Memory)  # bypass __init__
        mem.remember = fake_remember.__get__(mem)  # type: ignore
        bridge = LearnBridge(memory=mem, learning_dir=tmp_learning_dir, invoke_hermes=False)
        r = await bridge.learn("Hermes /learn rule: never log secrets to console")
        return r

    r = asyncio.run(run())
    assert r.chunk_id == "test-chunk-id"
    assert r.written_to is not None
    assert Path(r.written_to).exists()
    body = Path(r.written_to).read_text()
    assert "/learn" in body
    assert "never log secrets" in body


def test_learn_rejects_empty_text(tmp_path, tmp_learning_dir):
    """Empty text returns an error and does NOT write to disk."""
    from src.connectors.learn import LearnBridge
    from src.memory import Memory

    async def fake_remember(self, *a, **kw):
        raise AssertionError("remember should not be called for empty text")

    async def run():
        mem = Memory.__new__(Memory)
        mem.remember = fake_remember.__get__(mem)  # type: ignore
        bridge = LearnBridge(memory=mem, learning_dir=tmp_learning_dir, invoke_hermes=False)
        r = await bridge.learn("   ")
        return r

    r = asyncio.run(run())
    assert r.error == "empty text"
    # No files should exist in learning_dir.
    files = list(tmp_learning_dir.glob("*.md"))
    assert files == []


def test_learn_bridge_skips_hermes_when_disabled(tmp_path, tmp_learning_dir):
    """invoke_hermes=False means no shell-out even if hermes is on PATH."""
    from src.connectors.learn import LearnBridge
    from src.memory import Memory

    async def fake_remember(self, *a, **kw):
        return _fake_remember_result(chunk_id="x", importance=0.5)

    async def run():
        mem = Memory.__new__(Memory)
        mem.remember = fake_remember.__get__(mem)  # type: ignore
        bridge = LearnBridge(memory=mem, learning_dir=tmp_learning_dir, invoke_hermes=False)
        r = await bridge.learn("test rule")
        return r

    r = asyncio.run(run())
    assert r.hermes_invoked is False
    assert r.hermes_output == ""


# -----------------------------------------------------------------------------
# Track 3 — OpenClaw dreaming bridge
# -----------------------------------------------------------------------------


def _write_dream_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_dreaming_bridge_reads_diary(tmp_dreaming):
    """DREAMS.md entries are ingested into the brain as semantic tier."""
    from src.connectors.dreaming import DreamingBridge
    from src.memory import Memory, Tier

    diary = tmp_dreaming["diary"]
    _write_dream_file(diary, """<!-- openclaw:dreaming:diary:start -->

---

*April 11, 2026 at 7:00 AM*

There is a place where messages arrive like rain on a tin roof, and a small
machine decides whether they scatter shallow across stone or sink deep into
the soil.

---

*April 12, 2026 at 7:00 AM*

The router does not sleep. It only decides, again and again, who should carry
the weight.
""")

    remembered = []

    async def fake_remember(self, text, source_path, metadata=None, force_tier=None):
        remembered.append({"text": text, "source_path": source_path, "tier": force_tier})
        return _fake_remember_result(
            chunk_id=f"id-{len(remembered)}",
            importance=0.8,
        )

    async def run():
        # Build a minimal fake Memory
        mem = Memory.__new__(Memory)
        mem.remember = fake_remember.__get__(mem)  # type: ignore
        bridge = DreamingBridge(
            memory=mem,
            dreams_diary=tmp_dreaming["diary"],
            dreaming_dir=tmp_dreaming["dreaming_dir"],
            state_path=tmp_dreaming["state"],
        )
        return await bridge.read()

    r = asyncio.run(run())
    assert r.new_entries == 2
    assert r.by_kind.get("diary", 0) == 2
    # All ingested entries should be semantic tier
    for entry in remembered:
        assert entry["tier"] == Tier.SEMANTIC or entry["tier"] == "semantic"


def test_dreaming_bridge_reads_deep_dir(tmp_dreaming):
    """memory/dreaming/deep/*.md files are also ingested."""
    from src.connectors.dreaming import DreamingBridge
    from src.memory import Memory, Tier

    deep_file = tmp_dreaming["dreaming_dir"] / "deep" / "2026-05-07.md"
    _write_dream_file(deep_file, """# Dream 2026-05-07

The brain is a network of paths. Some lead to forgotten places; some lead
to a quiet room with a desk and a single light.
""")

    remembered = []

    async def fake_remember(self, text, source_path, metadata=None, force_tier=None):
        remembered.append({"text": text, "source_path": source_path, "tier": force_tier})
        return _fake_remember_result(chunk_id="x", importance=0.7)

    async def run():
        mem = Memory.__new__(Memory)
        mem.remember = fake_remember.__get__(mem)  # type: ignore
        bridge = DreamingBridge(
            memory=mem,
            dreams_diary=tmp_dreaming["diary"],
            dreaming_dir=tmp_dreaming["dreaming_dir"],
            state_path=tmp_dreaming["state"],
        )
        return await bridge.read()

    r = asyncio.run(run())
    assert r.new_entries >= 1
    assert r.by_kind.get("deep", 0) >= 1


def test_dreaming_bridge_is_idempotent(tmp_dreaming):
    """Running read() twice should not re-ingest the same entries."""
    from src.connectors.dreaming import DreamingBridge
    from src.memory import Memory, Tier

    diary = tmp_dreaming["diary"]
    _write_dream_file(diary, """<!-- openclaw:dreaming:diary:start -->

---

*April 11, 2026*

The router does not sleep. It only decides, again and again, who should
carry the weight of every incoming message.
""")

    remembered_count = {"n": 0}

    async def fake_remember(self, text, source_path, metadata=None, force_tier=None):
        remembered_count["n"] += 1
        return _fake_remember_result(chunk_id="x", importance=0.7)

    async def run():
        mem = Memory.__new__(Memory)
        mem.remember = fake_remember.__get__(mem)  # type: ignore
        bridge = DreamingBridge(
            memory=mem,
            dreams_diary=tmp_dreaming["diary"],
            dreaming_dir=tmp_dreaming["dreaming_dir"],
            state_path=tmp_dreaming["state"],
        )
        r1 = await bridge.read()
        r2 = await bridge.read()
        return r1, r2

    r1, r2 = asyncio.run(run())
    assert r1.new_entries == 1
    assert r2.new_entries == 0  # second call is a no-op
    assert r2.skipped == 1


class _FakeRecallResult:
    """Stand-in for a recall hit. Has .text, .tier, .metadata, .chunk_id, .importance."""
    def __init__(self, text, importance, tier="episodic", chunk_id="c"):
        self.text = text
        self.chunk_id = chunk_id
        self.tier = tier
        self.importance = importance
        self.metadata = {"importance": importance, "tier": tier}


class _FakeStats:
    """Minimal stand-in for QueryStats."""
    duration_seconds = 0.001


def _fake_recall(results):
    """Return the (results, stats) tuple shape that Memory.recall() uses."""
    return (results, _FakeStats())


def test_dreaming_bridge_cycle_writes_output_file(tmp_dreaming):
    """dreaming_cycle() writes a new entry to memory/dreaming/deep/<date>.md."""
    from src.connectors.dreaming import DreamingBridge
    from src.memory import Memory

    class FakeMemory:
        async def recall(self, query, k, tier=None, **kwargs):
            # Return 3 high-importance chunks for the requested tier. If
            # called twice (episodic + procedural), return same data; the
            # cycle deduplicates by chunk_id, so we'll still get 3.
            return _fake_recall([
                _FakeRecallResult(
                    text=f"high-importance chunk {i}: a durable fact about the system",
                    importance=0.8,
                    tier=tier or "episodic",
                    chunk_id=f"c{i}",
                )
                for i in range(3)
            ])

    async def run():
        mem = FakeMemory()
        bridge = DreamingBridge(
            memory=mem,
            dreams_diary=tmp_dreaming["diary"],
            dreaming_dir=tmp_dreaming["dreaming_dir"],
            state_path=tmp_dreaming["state"],
        )
        return await bridge.cycle(k=5, min_importance=0.5)

    r = asyncio.run(run())
    assert r.distilled_chunks == 3
    assert len(r.output_files) == 1
    out_path = Path(r.output_files[0])
    assert out_path.exists()
    body = out_path.read_text()
    assert "high-importance chunk" in body


def test_dreaming_bridge_cycle_skips_low_importance(tmp_dreaming):
    """Chunks below the min_importance threshold are excluded."""
    from src.connectors.dreaming import DreamingBridge

    class FakeMemory:
        async def recall(self, query, k, tier=None, **kwargs):
            return _fake_recall([
                _FakeRecallResult(
                    text="low importance chunk with enough text to pass the 40-char filter",
                    importance=0.1,
                    tier="episodic",
                    chunk_id="low",
                ),
                _FakeRecallResult(
                    text="high importance chunk with enough text to pass the 40-char filter",
                    importance=0.9,
                    tier="episodic",
                    chunk_id="high",
                ),
            ])

    async def run():
        mem = FakeMemory()
        bridge = DreamingBridge(
            memory=mem,
            dreams_diary=tmp_dreaming["diary"],
            dreaming_dir=tmp_dreaming["dreaming_dir"],
            state_path=tmp_dreaming["state"],
        )
        return await bridge.cycle(k=5, min_importance=0.5)

    r = asyncio.run(run())
    assert r.distilled_chunks == 1  # only the high-importance one


# -----------------------------------------------------------------------------
# Track 4 — Active Memory tool aliases
# -----------------------------------------------------------------------------


def test_active_memory_dispatch_unknown_tool():
    """Unknown tool names return ok=False with an error message."""
    from src.connectors.active_memory import ActiveMemoryAdapter, Brain

    # We don't need a real brain here — the dispatch layer short-circuits.
    adapter = ActiveMemoryAdapter.__new__(ActiveMemoryAdapter)
    adapter.brain = None  # type: ignore
    r = adapter.call("memory_query", {})
    # It WILL try to call brain.recall and fail because brain is None.
    # But the unknown-tool short-circuit must not fire because memory_query IS known.
    assert r["ok"] is False
    assert r["tool"] == "memory_query"
    assert "error" in r


def test_active_memory_unknown_tool_short_circuits():
    """An unknown tool returns ok=False without calling any brain method."""
    from src.connectors.active_memory import ActiveMemoryAdapter

    class ExplodingBrain:
        def __getattr__(self, name):
            raise AssertionError(f"brain.{name} should NOT be called for unknown tools")

    adapter = ActiveMemoryAdapter.__new__(ActiveMemoryAdapter)
    adapter.brain = ExplodingBrain()
    r = adapter.call("totally_not_a_real_tool", {"foo": "bar"})
    assert r["ok"] is False
    assert "unknown active-memory tool" in r["error"]


def test_active_memory_memory_store_force_tier_string():
    """memory_store accepts tier as a string (the v0.10.1 coercion fix)."""
    from src.connectors.active_memory import ActiveMemoryAdapter

    captured = {}

    class FakeBrain:
        def remember(self, **kwargs):
            captured.update(kwargs)
            # Real contract (v0.10.1+): Brain.remember() returns RememberResult,
            # NOT a bare string. The previous test stub simulated the old buggy
            # shape and pinned against the wrong code.
            from src.connectors.base import RememberResult
            return RememberResult(chunk_id="fake-chunk-id", tier="procedural", stored=True)

    adapter = ActiveMemoryAdapter.__new__(ActiveMemoryAdapter)
    adapter.brain = FakeBrain()
    r = adapter.memory_store(
        text="test fact",
        source="<test>",
        tier="procedural",  # string, not Tier enum
        metadata={"foo": "bar"},
    )
    assert r["ok"] is True
    assert r["data"]["chunk_id"] == "fake-chunk-id"
    # The Brain.remember wrapper handles coercion; we just verify the args passed.
    assert captured["force_tier"] == "procedural"
    assert captured["text"] == "test fact"


def test_active_memory_memory_query_dispatches_to_brain():
    """memory_query translates to brain.recall and returns the right shape."""
    from src.connectors.active_memory import ActiveMemoryAdapter
    from src.connectors.base import Brain

    class FakeResult:
        chunk_id = "c1"
        text = "result text"
        tier = "semantic"
        score = 0.9
        metadata = {"source_path": "/test.md"}

    # Brain.recall() returns a plain list[RecallResult], not an object
    # with a .results attribute.
    class FakeBrain:
        def recall(self, **kwargs):
            assert kwargs["query"] == "hello"
            return [FakeResult()]

    adapter = ActiveMemoryAdapter.__new__(ActiveMemoryAdapter)
    adapter.brain = FakeBrain()
    r = adapter.memory_query(query="hello", k=3)
    assert r["ok"] is True
    assert r["tool"] == "memory_query"
    assert len(r["data"]["results"]) == 1
    assert r["data"]["results"][0]["text"] == "result text"


def test_active_memory_memory_forget_uses_forget_by_query():
    """memory_forget calls brain.forget_by_query."""
    from src.connectors.active_memory import ActiveMemoryAdapter

    captured = {}

    class FakeBrain:
        def forget_by_query(self, **kwargs):
            captured.update(kwargs)
            return 2  # number removed

    adapter = ActiveMemoryAdapter.__new__(ActiveMemoryAdapter)
    adapter.brain = FakeBrain()
    r = adapter.memory_forget(query="duckets token", k=5)
    assert r["ok"] is True
    assert r["data"]["removed"] == 2
    assert captured["query"] == "duckets token"
    assert captured["k"] == 5


def test_active_memory_call_dispatches_by_name():
    """adapter.call(tool, args) dispatches to the right method."""
    from src.connectors.active_memory import ActiveMemoryAdapter

    captured = {}

    class FakeBrain:
        def recall(self, **kwargs):
            captured["recall"] = kwargs
            from src.connectors.base import Brain
            # Brain.recall() returns list[RecallResult] (a plain list).
            return []
        def forget_by_query(self, **kwargs):
            captured["forget"] = kwargs
            return 0
        def remember(self, **kwargs):
            captured["remember"] = kwargs
            return "id"

    adapter = ActiveMemoryAdapter.__new__(ActiveMemoryAdapter)
    adapter.brain = FakeBrain()
    r1 = adapter.call("memory_query", {"query": "x", "k": 3})
    assert r1["ok"] is True and r1["tool"] == "memory_query"
    r2 = adapter.call("memory_forget", {"query": "y", "k": 2})
    assert r2["ok"] is True and r2["tool"] == "memory_forget"
    r3 = adapter.call("memory_store", {"text": "z"})
    assert r3["ok"] is True and r3["tool"] == "memory_store"
    assert "remember" in captured


# -----------------------------------------------------------------------------
# Track 5 — Brain facade exposes the v0.11.0 methods
# -----------------------------------------------------------------------------


def test_brain_facade_has_v0_11_methods():
    """The Brain facade must expose dreaming_read, dreaming_cycle, learn,
    and active_memory methods (track 1 wiring check)."""
    from src.connectors.base import Brain

    brain = Brain.__new__(Brain)  # don't init memory yet
    for method in ("dreaming_read", "dreaming_cycle", "learn", "active_memory"):
        assert hasattr(brain, method), f"Brain missing method: {method}"
        assert callable(getattr(brain, method)), f"Brain.{method} not callable"


# -----------------------------------------------------------------------------
# MCP server tool registration
# -----------------------------------------------------------------------------


def test_mcp_server_has_v0_11_tools():
    """The MCP server TOOLS list must include the v0.11.0 tools."""
    from src.mcp_server import TOOLS

    tool_names = {t["name"] for t in TOOLS}
    for required in ("dreaming_read", "dreaming_cycle", "learn", "active_memory"):
        assert required in tool_names, f"MCP server missing tool: {required}"


def test_mcp_server_has_handlers_for_v0_11():
    """The MCP server HANDLERS dict must dispatch the v0.11.0 tools."""
    from src.mcp_server import HANDLERS

    for required in ("dreaming_read", "dreaming_cycle", "learn", "active_memory"):
        assert required in HANDLERS, f"MCP server HANDLERS missing: {required}"
        assert callable(HANDLERS[required])


# -----------------------------------------------------------------------------
# OpenClaw connector tool registration
# -----------------------------------------------------------------------------


def test_openclaw_connector_has_v0_11_tools():
    """The OpenClaw connector's TOOL_DEFINITIONS must include the v0.11.0 tools."""
    from src.connectors.openclaw import TOOL_DEFINITIONS

    tool_names = {t["name"] for t in TOOL_DEFINITIONS}
    for required in (
        "brain_dreaming_read",
        "brain_dreaming_cycle",
        "brain_learn",
        "brain_active_memory",
    ):
        assert required in tool_names, f"OpenClaw connector missing: {required}"
