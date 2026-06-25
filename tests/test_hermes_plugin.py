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


def test_get_tool_schemas_returns_eleven_tools():
    """v0.14.0: surface expanded to 11 tools (incl. brain_wake_up +
    brain_skills_list/promote for the agent-driven skill pipeline).
    Same list as the OpenClaw extension adapter."""
    p = DuckBotBrainProvider()
    schemas = p.get_tool_schemas()
    assert len(schemas) == 11
    names = {s["function"]["name"] for s in schemas}
    expected = {
        "brain_wake_up", "brain_recall", "brain_recall_verbatim",
        "brain_remember", "brain_reflect", "brain_stats",
        "brain_fsrs_review", "brain_decay_status", "brain_search_verbatim",
        "brain_skills_list", "brain_skills_promote",
    }
    assert names == expected


def test_brain_wake_up_schema_includes_session_start_args():
    """brain_wake_up must take query/k/include_blocks/etc."""
    p = DuckBotBrainProvider()
    schemas = p.get_tool_schemas()
    wake = next(s for s in schemas if s["function"]["name"] == "brain_wake_up")
    props = wake["function"]["parameters"]["properties"]
    assert "query" in props
    assert "k" in props
    assert "include_blocks" in props
    assert "include_graph" in props
    assert "include_fsrs_review" in props


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
    """v0.14.0: brain_reflect now delegates to the real Memory.reflect()
    pipeline (sleep-time episodic → semantic consolidation). The shape
    is the consolidation summary, not a snippets wrap."""
    provider._brain = fake_brain
    # Memory.reflect is async; the dispatch layer calls it via _run_async
    # which itself spawns a worker thread. We mock the path so the test
    # doesn't hit Chroma.
    fake_consolidation = {
        "scanned": 200, "extracted": 64,
        "after_dedup": 47, "promoted": 47,
    }
    import src.extensions.tools as tools_mod
    with patch.object(tools_mod, "dispatch", return_value=fake_consolidation) as mock_disp:
        out = provider.handle_tool_call("brain_reflect", {"query": "test"})
    data = json.loads(out)
    assert "scanned" in data
    assert "promoted" in data
    mock_disp.assert_called_once_with("brain_reflect", {"query": "test"})


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
    """plugin.yaml must list the hooks we implement.

    v0.14.0: both on_session_start AND on_session_end. The previous
    version only listed on_session_end (a no-op stub)."""
    yaml_file = (
        Path(__file__).resolve().parent.parent
        / "src" / "plugins" / "memory" / "duckbot_brain" / "plugin.yaml"
    )
    content = yaml_file.read_text()
    assert "on_session_end" in content
    assert "on_session_start" in content


def test_on_session_end_no_messages_returns_none(provider):
    """No messages → nothing to consolidate → return None."""
    assert provider.on_session_end([]) is None
    assert provider.on_session_end(None) is None


def test_on_session_end_skips_non_primary_context(provider):
    """Per Hermes ABC: skip the procedural write for cron/subagent/flush."""
    provider.initialize("s1", agent_context="cron")
    messages = [{"role": "user", "content": "I always want compact output"}]
    out = provider.on_session_end(messages)
    assert out == {"skipped": "non-primary context"}


def test_on_session_end_persists_durable_user_rules(provider, fake_brain):
    """Durable-shaped user statements (always/never/prefer/...) get
    collected and queued for remember() as one procedural chunk."""
    provider.initialize("s1", agent_context="primary", platform="cli")
    provider._brain = fake_brain
    # Replace the executor with a mock so we can inspect the submission
    # without actually running it.
    fake_executor = MagicMock()
    provider._executor = fake_executor

    messages = [
        {"role": "user", "content": "I always want compact output for crons."},
        {"role": "user", "content": "Don't ask the wallet — read the file directly."},
        {"role": "assistant", "content": "Got it, switching to compact format."},
        {"role": "user", "content": "what's the weather?"},  # not durable-shaped
    ]
    out = provider.on_session_end(messages)
    assert out is not None
    assert out["persisted"] >= 2
    assert out["tier"] == "procedural"
    assert "hermes://session-end/" in out["source"]
    fake_executor.submit.assert_called_once()
    # Args: (callable, text, source, tier)
    args = fake_executor.submit.call_args[0]
    assert args[3] == "procedural"
    assert "hermes://session-end/" in args[2]
    # The procedural chunk text contains the durable rules.
    assert "always want compact output" in args[1]
    assert "Don't ask the wallet" in args[1]


def test_on_session_end_no_durable_statements_returns_none(provider, fake_brain):
    """Session with no durable-shaped user statements → return None,
    no executor submit."""
    provider.initialize("s1", agent_context="primary")
    provider._brain = fake_brain
    fake_executor = MagicMock()
    provider._executor = fake_executor

    messages = [
        {"role": "user", "content": "what's 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "ok thanks"},
    ]
    out = provider.on_session_end(messages)
    assert out is None
    fake_executor.submit.assert_not_called()


def test_on_session_start_returns_wake_up_shape(provider, fake_brain):
    """on_session_start returns the brain_wake_up dict (memories +
    blocks + graph + fsrs queue + stats) so callers can pre-load
    context without an extra MCP round-trip."""
    fake_brain.wake_up.return_value = {
        "memories": [], "blocks": [],
        "graph_summary": {}, "fsrs_review_queue": [], "stats": {},
    }
    provider._brain = fake_brain
    out = provider.on_session_start()
    assert out is not None
    assert "memories" in out
    fake_brain.wake_up.assert_called_once()


def test_plugin_init_uses_expected_imports():
    """The plugin must import Brain from src.connectors.base (the integration seam)."""
    init = (
        Path(__file__).resolve().parent.parent
        / "src" / "plugins" / "memory" / "duckbot_brain" / "__init__.py"
    )
    content = init.read_text()
    assert "from src.connectors.base import Brain" in content


def test_on_session_end_captures_short_preferences(provider, fake_brain):
    """Short durable preferences (12-30 chars) MUST be captured — the
    old 30-char floor rejected real preferences like 'I always prefer dark mode'."""
    provider.initialize("s1", agent_context="primary")
    provider._brain = fake_brain
    fake_executor = MagicMock()
    provider._executor = fake_executor

    messages = [
        {"role": "user", "content": "I always prefer dark mode"},  # 25 chars
        {"role": "user", "content": "I never use tabs"},            # 16 chars
    ]
    out = provider.on_session_end(messages)
    assert out is not None, "short preferences must not be silently dropped"
    assert out["persisted"] == 2
    fake_executor.submit.assert_called_once()


def test_on_session_end_rejects_ultra_short_noise(provider, fake_brain):
    """Ultra-short fragments (< 12 chars) like 'ok always' are noise
    (mid-conversation acks), not durable rules — must still be rejected."""
    provider.initialize("s1", agent_context="primary")
    provider._brain = fake_brain
    fake_executor = MagicMock()
    provider._executor = fake_executor

    messages = [
        {"role": "user", "content": "ok always"},   # 9 chars
        {"role": "user", "content": "yes never"},   # 9 chars
    ]
    out = provider.on_session_end(messages)
    assert out is None
    fake_executor.submit.assert_not_called()
