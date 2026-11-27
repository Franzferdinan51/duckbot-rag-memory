"""
test_v0_11_integration.py — tests for the v0.11.0 integrations:
  - Track 2: Hermes /learn shim
  - Track 5: Brain facade + MCP registration wiring

(Tracks 3 and 4 — the OpenClaw dreaming bridge and Active Memory
aliases — were removed in v0.19.0 with the rest of OpenClaw support;
the facade/registration tests now assert their absence.)

We don't shell out to `hermes` in tests.
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


def test_learn_bridge_ingests_to_brain(tmp_path):
    """Hermes /learn shim ingests into the brain (v0.19.0: the old
    OpenClaw learning-dir file write is gone — written_to stays None)."""
    from src.connectors.learn import LearnBridge
    from src.memory import Memory

    # We can't run the real Memory here (needs LM Studio), so we test the
    # file-writing side directly by patching the memory call.
    async def fake_remember(self, *a, **kw):
        return _fake_remember_result(chunk_id="test-chunk-id")

    async def run():
        mem = Memory.__new__(Memory)  # bypass __init__
        mem.remember = fake_remember.__get__(mem)  # type: ignore
        bridge = LearnBridge(memory=mem, invoke_hermes=False)
        r = await bridge.learn("Hermes /learn rule: never log secrets to console")
        return r

    r = asyncio.run(run())
    assert r.chunk_id == "test-chunk-id"
    assert r.error is None
    assert r.written_to is None


def test_learn_rejects_empty_text(tmp_path):
    """Empty text returns an error and never reaches the brain."""
    from src.connectors.learn import LearnBridge
    from src.memory import Memory

    async def fake_remember(self, *a, **kw):
        raise AssertionError("remember should not be called for empty text")

    async def run():
        mem = Memory.__new__(Memory)
        mem.remember = fake_remember.__get__(mem)  # type: ignore
        bridge = LearnBridge(memory=mem, invoke_hermes=False)
        r = await bridge.learn("   ")
        return r

    r = asyncio.run(run())
    assert r.error == "empty text"
    assert r.chunk_id is None


def test_learn_bridge_skips_hermes_when_disabled(tmp_path):
    """invoke_hermes=False means no shell-out even if hermes is on PATH."""
    from src.connectors.learn import LearnBridge
    from src.memory import Memory

    async def fake_remember(self, *a, **kw):
        return _fake_remember_result(chunk_id="x", importance=0.5)

    async def run():
        mem = Memory.__new__(Memory)
        mem.remember = fake_remember.__get__(mem)  # type: ignore
        bridge = LearnBridge(memory=mem, invoke_hermes=False)
        r = await bridge.learn("test rule")
        return r

    r = asyncio.run(run())
    assert r.hermes_invoked is False
    assert r.hermes_output == ""


# -----------------------------------------------------------------------------
# Track 5 — Brain facade exposes the v0.11.0 methods
# -----------------------------------------------------------------------------


def test_brain_facade_has_v0_11_methods():
    """The Brain facade must expose learn (track 1 wiring check).
    v0.19.0: dreaming_read / dreaming_cycle / active_memory were
    removed with OpenClaw support and must stay gone."""
    from src.connectors.base import Brain

    brain = Brain.__new__(Brain)  # don't init memory yet
    assert hasattr(brain, "learn") and callable(brain.learn)
    for method in ("dreaming_read", "dreaming_cycle", "active_memory"):
        assert not hasattr(brain, method), f"Brain must not expose removed method: {method}"


# -----------------------------------------------------------------------------
# MCP server tool registration
# -----------------------------------------------------------------------------


def test_mcp_server_has_v0_11_tools():
    """The MCP server TOOLS list must include the v0.11.0 tools.

    v0.13.x: this test now asserts dynamically — no hardcoded count.
    The v0.11.0 tool contract is preserved (the four tools below must
    still exist), but new tools added since v0.11.0 don't break the test.
    """
    from src.mcp_server import TOOLS

    tool_names = {t["name"] for t in TOOLS}
    assert "learn" in tool_names, "MCP server missing tool: learn"
    for removed in ("dreaming_read", "dreaming_cycle", "active_memory"):
        assert removed not in tool_names, f"removed tool still registered: {removed}"
    # Dynamic invariant: every TOOL must have a matching HANDLER entry.
    from src.mcp_server import HANDLERS
    tool_set = tool_names
    handler_set = set(HANDLERS.keys())
    assert tool_set == handler_set, (
        f"Tool/handler mismatch: only in TOOLS = {tool_set - handler_set}; "
        f"only in HANDLERS = {handler_set - tool_set}"
    )
    # And the count must be at least the v0.11.0 floor.
    assert len(TOOLS) >= 4, f"MCP server only exposes {len(TOOLS)} tools"


def test_mcp_server_has_handlers_for_v0_11():
    """The MCP server HANDLERS dict must dispatch the v0.11.0 tools."""
    from src.mcp_server import HANDLERS

    assert "learn" in HANDLERS and callable(HANDLERS["learn"])
    for removed in ("dreaming_read", "dreaming_cycle", "active_memory"):
        assert removed not in HANDLERS, f"removed handler still registered: {removed}"


def test_brain_remember_and_brain_recall_in_mcp_server():
    """brain_remember and brain_recall must be registered in both TOOLS and HANDLERS.

    Regression: these were in brain_tools.TOOL_DEFINITIONS but were never added to
    the MCP server's own TOOLS list or HANDLERS dict, causing direct MCP calls
    to fail with 'coroutine was expected' or 'NoneType is not callable'.
    """
    from src.mcp_server import TOOLS, HANDLERS

    tool_names = {t["name"] for t in TOOLS}
    for name in ("brain_remember", "brain_recall"):
        assert name in tool_names, f"brain_* alias not in MCP TOOLS: {name}"
        assert name in HANDLERS, f"brain_* alias not in MCP HANDLERS: {name}"
        assert callable(HANDLERS[name])


# -----------------------------------------------------------------------------
# brain_tools connector tool registration
# -----------------------------------------------------------------------------


def test_brain_tools_connector_has_v0_11_tools():
    """brain_tools TOOL_DEFINITIONS must include brain_learn (v0.11.0);
    the dreaming/active-memory tools were removed in v0.19.0."""
    from src.connectors.brain_tools import TOOL_DEFINITIONS

    tool_names = {t["name"] for t in TOOL_DEFINITIONS}
    assert "brain_learn" in tool_names
    for removed in ("brain_dreaming_read", "brain_dreaming_cycle", "brain_active_memory"):
        assert removed not in tool_names, f"removed tool still defined: {removed}"
