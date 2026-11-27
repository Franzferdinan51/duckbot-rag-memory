"""
test_reconciliation_v019.py — guards for the v0.19.0 OpenClaw removal.

Owner's direction (2026-11-26): "we don't really support OpenClaw
anymore because I use you and Hermes." These tests pin the removal so
the surface can't silently grow back: no openclaw module, no dreaming /
active-memory tools anywhere in the live surface, Hermes-only sync,
and the embedder load-on-call self-heal that fixed the wedged
"Running brain wake up" (a cold embedder made wake_up take ~8.6s,
over Hermes' 8s prefetch timeout).
"""

from __future__ import annotations

import importlib.util

import pytest


def test_openclaw_connector_module_is_gone():
    assert importlib.util.find_spec("src.connectors.openclaw") is None
    assert importlib.util.find_spec("src.connectors.openclaw_shim") is None
    assert importlib.util.find_spec("src.connectors.dreaming") is None
    assert importlib.util.find_spec("src.connectors.active_memory") is None


def test_brain_tools_module_is_the_connector_layer():
    from src.connectors.brain_tools import TOOL_DEFINITIONS, handle

    names = {t["name"] for t in TOOL_DEFINITIONS}
    assert "brain_learn" in names
    for removed in ("brain_dreaming_read", "brain_dreaming_cycle", "brain_active_memory"):
        assert removed not in names
    assert callable(handle)


def test_brain_facade_has_no_removed_methods():
    from src.connectors.base import Brain

    brain = Brain.__new__(Brain)
    assert callable(brain.learn)
    for method in ("dreaming_read", "dreaming_cycle", "active_memory"):
        assert not hasattr(brain, method)


def test_mcp_surface_has_no_removed_tools():
    from src.mcp_server import HANDLERS, TOOLS

    names = {t["name"] for t in TOOLS}
    for removed in (
        "dreaming_read", "dreaming_cycle", "active_memory",
        "brain_dreaming_read", "brain_dreaming_cycle", "brain_active_memory",
    ):
        assert removed not in names
        assert removed not in HANDLERS
    # every listed tool still has a handler (registration integrity)
    assert names == set(HANDLERS.keys())


@pytest.mark.asyncio
async def test_sync_rejects_openclaw_target():
    from src.mcp_server import handle_brain_sync

    r = await handle_brain_sync({"target": "openclaw", "dry_run": True})
    assert "error" in r
    assert "Hermes only" in r["error"]


def test_embedder_loads_on_call():
    """The LM Studio embedder must self-heal a TTL-unloaded model."""
    from src.embeddings import LMStudioEmbeddings

    assert callable(getattr(LMStudioEmbeddings, "_ensure_loaded", None))
