"""
test_hermes_plugin.py — verify the duckbot_brain Hermes MemoryProvider plugin.

Pattern: import the plugin module (which is self-contained) and exercise its
duckbot-secret-scan: allowlist-file
methods against a stub Brain so we don't hit Chroma during tests.
"""

# duckbot-secret-scan: allowlist-file
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Make `src` importable from the tests/ dir.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plugins.memory.duckbot_brain import (  # noqa: E402
    DuckBotBrainProvider,
    register,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def provider() -> DuckBotBrainProvider:
    return DuckBotBrainProvider()


@pytest.fixture
def fake_brain():
    """A MagicMock that looks enough like Brain for the provider to call."""
    brain = MagicMock()
    brain.recall.return_value = [
        MagicMock(
            chunk_id="c1",
            text="Sample chunk text.",
            tier="semantic",
            importance=0.7,
            score=0.42,
            source_path="/tmp/test.md",
            metadata={"source_path": "/tmp/test.md"},
        ),
        MagicMock(
            chunk_id="c2",
            text="Another chunk.",
            tier="episodic",
            importance=0.3,
            score=0.18,
            source_path="/tmp/other.md",
            metadata={"source_path": "/tmp/other.md"},
        ),
    ]
    brain.recall_verbatim.return_value = [
        {"chunk_id": "c1", "verbatim_text": "original text"},
    ]
    return brain


# -----------------------------------------------------------------------------
# Identity & lifecycle
# -----------------------------------------------------------------------------


def test_provider_name():
    p = DuckBotBrainProvider()
    assert p.name == "duckbot-brain"


def test_is_available_returns_true():
    p = DuckBotBrainProvider()
    assert p.is_available() is True


def test_initialize_stores_session_metadata():
    p = DuckBotBrainProvider()
    p.initialize(
        session_id="sess-1",
        hermes_home="/tmp/hermes",
        platform="telegram",
        agent_context="primary",
    )
    assert p._session_id == "sess-1"
    assert p._hermes_home == "/tmp/hermes"
    assert p._platform == "telegram"
    assert p._agent_context == "primary"


def test_shutdown_is_safe_to_call():
    p = DuckBotBrainProvider()
    p.shutdown()
    # Calling twice should not raise.
    p.shutdown()


# -----------------------------------------------------------------------------
# register()
# -----------------------------------------------------------------------------


def test_register_via_ctx_with_register_memory_provider():
    """The standard Hermes plugin context shape."""
    captured = {}
    ctx = MagicMock()
    ctx.register_memory_provider = lambda provider: captured.setdefault("p", provider)

    register(ctx)
    assert "p" in captured
    assert isinstance(captured["p"], DuckBotBrainProvider)


def test_register_via_ctx_with_flat_register_callable():
    """Some plugin loaders use a flat register(provider)."""
    captured = []
    ctx = lambda provider: captured.append(provider)

    register(ctx)
    assert len(captured) == 1
    assert isinstance(captured[0], DuckBotBrainProvider)


# -----------------------------------------------------------------------------
# system_prompt_block
# -----------------------------------------------------------------------------


def test_system_prompt_block_mentions_brain_tools():
    p = DuckBotBrainProvider()
    block = p.system_prompt_block()
    assert "brain_recall" in block
    assert "brain_recall_verbatim" in block
    assert "brain_reflect" in block


# -----------------------------------------------------------------------------
# prefetch
# -----------------------------------------------------------------------------


def test_prefetch_returns_empty_for_empty_query():
    p = DuckBotBrainProvider()
    assert p.prefetch("") == ""


def test_prefetch_returns_formatted_context(provider, fake_brain):
    provider._brain = fake_brain
    out = provider.prefetch("test query")
    assert out.startswith("[memory]")
    # Both chunks appear in the output.
    assert "Sample chunk text." in out
    assert "Another chunk." in out
    # Source paths annotated.
    assert "/tmp/test.md" in out


def test_prefetch_truncates_long_chunks(provider, fake_brain):
    """Chunks over 240 chars get truncated with '...'."""
    fake_brain.recall.return_value[0].text = "x" * 1000
    provider._brain = fake_brain
    out = provider.prefetch("test")
    assert "..." in out


def test_prefetch_handles_recall_failure(provider):
    """If recall throws, prefetch returns empty (Hermes contract: must be fast)."""
    provider._brain = MagicMock()
    provider._brain.recall.side_effect = RuntimeError("boom")
    assert provider.prefetch("test") == ""


def test_prefetch_handles_empty_results(provider, fake_brain):
    fake_brain.recall.return_value = []
    provider._brain = fake_brain
    assert provider.prefetch("test") == ""


# -----------------------------------------------------------------------------
# sync_turn
# -----------------------------------------------------------------------------


def test_sync_turn_skips_non_primary_context(provider):
    """Per Hermes ABC: skip writes for cron/subagent/flush contexts."""
    provider.initialize("s1", agent_context="cron")
    # Should NOT call remember().
    provider._brain = MagicMock()
    provider.sync_turn("hi", "hello there")
    provider._brain.remember.assert_not_called()


def test_sync_turn_skips_empty_inputs(provider):
    """Don't remember empty user/assistant content."""
    provider.initialize("s1", agent_context="primary")
    provider._brain = MagicMock()
    provider.sync_turn("", "")
    provider.sync_turn("", "hello")
    provider.sync_turn("hi", "")
    provider._brain.remember.assert_not_called()


def test_sync_turn_queues_background_executor(provider):
    """sync_turn must be non-blocking — submits to executor."""
    provider.initialize("s1", agent_context="primary", platform="cli")
    provider._brain = MagicMock()
    # Patch the executor to a MagicMock so we can verify submission.
    fake_executor = MagicMock()
    provider._executor = fake_executor

    provider.sync_turn("user said hi", "assistant replied")

    fake_executor.submit.assert_called_once()
    args = fake_executor.submit.call_args[0]
    # First arg is the callable; second/third are user/assistant content.
    assert callable(args[0])
    assert args[1] == "user said hi"
    assert args[2] == "assistant replied"


def test_sync_turn_blocking_calls_remember(provider, fake_brain):
    """The blocking version actually invokes brain.remember()."""
    import asyncio
    provider.initialize("s1", agent_context="primary", platform="telegram")
    provider._brain = fake_brain
    # Bypass asyncio.run since MagicMock.remember is sync; the plugin
    # calls asyncio.run(brain.remember(...)). Patch asyncio to avoid
    # event-loop issues in test.
    with patch("asyncio.run") as mock_run:
        provider._sync_turn_blocking("hi", "hello")
        mock_run.assert_called_once()


# -----------------------------------------------------------------------------
# Tools: get_tool_schemas
# -----------------------------------------------------------------------------


def test_get_tool_schemas_returns_three_tools():
    p = DuckBotBrainProvider()
    schemas = p.get_tool_schemas()
    assert len(schemas) == 3
    names = {s["function"]["name"] for s in schemas}
    assert names == {"brain_recall", "brain_recall_verbatim", "brain_reflect"}


def test_brain_recall_schema_includes_rerank_and_decay():
    p = DuckBotBrainProvider()
    schemas = p.get_tool_schemas()
    recall = next(s for s in schemas if s["function"]["name"] == "brain_recall")
    props = recall["function"]["parameters"]["properties"]
    assert "query" in props
    assert "rerank" in props
    assert "decay" in props
    assert "query" in recall["function"]["parameters"]["required"]


# -----------------------------------------------------------------------------
# Tools: handle_tool_call
# -----------------------------------------------------------------------------


def test_handle_tool_call_brain_recall(provider, fake_brain):
    provider._brain = fake_brain
    out = provider.handle_tool_call("brain_recall", {"query": "test", "k": 2, "rerank": True})
    data = json.loads(out)
    assert "results" in data
    assert len(data["results"]) == 2
    fake_brain.recall.assert_called_once_with(
        query="test", k=2, tier=None, min_importance=None, rerank=True, decay=False,
    )


def test_handle_tool_call_brain_recall_verbatim(provider, fake_brain):
    provider._brain = fake_brain
    out = provider.handle_tool_call(
        "brain_recall_verbatim", {"query": "test", "tier": "semantic"}
    )
    data = json.loads(out)
    assert "results" in data
    assert data["results"][0]["verbatim_text"] == "original text"
    fake_brain.recall_verbatim.assert_called_once()


def test_handle_tool_call_brain_reflect(provider, fake_brain):
    provider._brain = fake_brain
    out = provider.handle_tool_call("brain_reflect", {"query": "test"})
    data = json.loads(out)
    assert "snippets" in data
    assert "note" in data  # tells caller this is pre-LLM synthesis


def test_handle_tool_call_unknown_tool(provider, fake_brain):
    provider._brain = fake_brain
    out = provider.handle_tool_call("brain_does_not_exist", {})
    data = json.loads(out)
    assert "error" in data


def test_handle_tool_call_handles_failure(provider):
    provider._brain = MagicMock()
    provider._brain.recall.side_effect = RuntimeError("chroma down")
    out = provider.handle_tool_call("brain_recall", {"query": "test"})
    data = json.loads(out)
    assert "error" in data
    assert "chroma down" in data["error"]


# -----------------------------------------------------------------------------
# Discovery shape: can the Hermes plugin loader find us?
# -----------------------------------------------------------------------------


def test_plugin_yaml_present():
    """The plugin.yaml file must exist for Hermes discovery."""
    plugin_dir = Path(__file__).resolve().parent.parent / "src" / "plugins" / "memory" / "duckbot_brain"
    assert plugin_dir.is_dir(), f"plugin dir missing: {plugin_dir}"
    yaml_file = plugin_dir / "plugin.yaml"
    assert yaml_file.exists(), "plugin.yaml required for Hermes discovery"
    content = yaml_file.read_text()
    assert "name: duckbot-brain" in content


def test_plugin_init_contains_required_hooks():
    """plugin.yaml must list the hooks we implement."""
    yaml_file = (
        Path(__file__).resolve().parent.parent
        / "src" / "plugins" / "memory" / "duckbot_brain" / "plugin.yaml"
    )
    content = yaml_file.read_text()
    assert "on_session_end" in content


def test_plugin_init_uses_expected_imports():
    """The plugin must import Brain from src.connectors.base (the integration seam)."""
    init = (
        Path(__file__).resolve().parent.parent
        / "src" / "plugins" / "memory" / "duckbot_brain" / "__init__.py"
    )
    content = init.read_text()
    assert "from src.connectors.base import Brain" in content
