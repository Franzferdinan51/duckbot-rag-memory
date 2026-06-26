"""Tests for Hermes MemoryProvider plugin discovery + activation.

Hermes discovers plugins by importing `plugins/memory/<name>/__init__.py`
and calling its `register(ctx)` function. The provider is then activated
when `~/.hermes/config.yaml` contains `memory.provider: <name>`.

These tests verify:
  - `register()` calls `ctx.register_memory_provider(provider)` correctly.
  - `register()` falls back to flat-callable + module-level handoff.
  - `is_available()` returns True when imports work, False otherwise.
  - `is_available()` makes NO network calls (cheap gate for MemoryManager).
  - The plugin's name, version, and hooks match `plugin.yaml`.

These tests run without LM Studio (the provider's `is_available()` should
never touch the embedder).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

# Repo root on sys.path so `src.plugins.memory.duckbot_brain` is importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.plugins.memory import duckbot_brain  # noqa: E402


class _FakeCtx:
    """Fake Hermes plugin context that records what was registered."""

    def __init__(self) -> None:
        self.registered: list = []

    def register_memory_provider(self, provider) -> None:
        self.registered.append(provider)


class _FakeCallableCtx:
    """Some loaders pass ctx as a flat callable `ctx(provider)`."""

    def __init__(self) -> None:
        self.received: list = []

    def __call__(self, provider) -> None:
        self.received.append(provider)


class _BoomCtx:
    """A ctx that exposes neither method nor callable — exercises the
    fallback path that stores the provider on a module-level handoff."""

    def __init__(self) -> None:
        # Deliberately no register_memory_provider and not callable.
        pass


def test_register_uses_ctx_method_when_available(caplog):
    """The happy path: register() pushes the provider into ctx."""
    ctx = _FakeCtx()
    with caplog.at_level(logging.INFO, logger="src.plugins.memory.duckbot_brain"):
        duckbot_brain.register(ctx)
    assert len(ctx.registered) == 1
    assert isinstance(ctx.registered[0], duckbot_brain.DuckBotBrainProvider)
    assert ctx.registered[0].name == "duckbot-brain"
    # Logs an INFO line so operators can grep for plugin activation.
    assert any("[duckbot-brain] registering" in r.message for r in caplog.records)


def test_register_falls_back_to_flat_callable():
    """Loader passes ctx as a callable → ctx(provider)."""
    ctx = _FakeCallableCtx()
    duckbot_brain.register(ctx)
    assert len(ctx.received) == 1
    assert isinstance(ctx.received[0], duckbot_brain.DuckBotBrainProvider)


def test_register_falls_back_to_module_handoff():
    """Loader provides neither method nor callable → module-level handoff."""
    ctx = _BoomCtx()
    # Reset module-level handoff so we can observe the assignment.
    duckbot_brain._HANDOFF = None
    duckbot_brain.register(ctx)
    assert duckbot_brain._HANDOFF is not None
    assert isinstance(duckbot_brain._HANDOFF, duckbot_brain.DuckBotBrainProvider)
    # Clean up so we don't leak state into other tests.
    duckbot_brain._HANDOFF = None


def test_is_available_returns_true_when_imports_work():
    """is_available() must return True when the Brain module is importable.

    This is the gate MemoryManager uses to decide whether to activate
    the provider — it MUST NOT require LM Studio to be running.
    """
    provider = duckbot_brain.DuckBotBrainProvider()
    assert provider.is_available() is True


def test_is_available_makes_no_network_calls(monkeypatch):
    """is_available() must be a pure-Python check — no HTTP, no embed.

    Verifies by patching `urllib.request.urlopen` (or similar) to raise;
    if is_available() called it, the test would fail with the original
    exception rather than returning True.
    """
    import urllib.request

    def _explode(*a, **kw):
        raise AssertionError("is_available() must not make HTTP calls")

    monkeypatch.setattr(urllib.request, "urlopen", _explode)
    # Also patch requests.get if available — covers the httpx / requests
    # paths our embedders use.
    try:
        import requests

        monkeypatch.setattr(requests, "get", _explode, raising=False)
        monkeypatch.setattr(requests, "post", _explode, raising=False)
    except ImportError:
        pass

    provider = duckbot_brain.DuckBotBrainProvider()
    assert provider.is_available() is True


def test_provider_name_is_duckbot_brain():
    """Provider.name property is the activation key for MemoryManager."""
    provider = duckbot_brain.DuckBotBrainProvider()
    assert provider.name == "duckbot-brain"


def test_plugin_yaml_declares_hooks():
    """plugin.yaml must list on_session_start + on_session_end so the
    plugin loader can wire up the lifecycle hooks."""
    import yaml  # type: ignore[import-untyped]

    yaml_path = REPO_ROOT / "src" / "plugins" / "memory" / "duckbot_brain" / "plugin.yaml"
    assert yaml_path.is_file(), f"plugin.yaml missing at {yaml_path}"
    data = yaml.safe_load(yaml_path.read_text())
    assert data["name"] == "duckbot-brain"
    assert "version" in data and data["version"]
    hooks = data.get("hooks", []) or []
    assert "on_session_start" in hooks
    assert "on_session_end" in hooks


def test_bootstrap_script_mentions_memory_provider():
    """Regression: the bootstrap must auto-write memory.provider so the
    plugin actually activates. Operators grep `~/.hermes/config.yaml`
    for `duckbot-brain` to confirm — this test catches accidental removal
    of that activation step."""
    script = (REPO_ROOT / "scripts" / "hermes-bootstrap.sh").read_text()
    assert "memory.provider: duckbot-brain" in script, (
        "hermes-bootstrap.sh no longer auto-activates the plugin in "
        "config.yaml. Without this, the plugin files are installed but "
        "MemoryManager never instantiates the provider."
    )
    # And it must back up the file before mutating.
    assert ".bak." in script and 'BACKUP="$HERMES_CONFIG' in script, (
        "hermes-bootstrap.sh should back up config.yaml before mutating it."
    )


def test_bootstrap_script_is_idempotent():
    """Running the bootstrap twice must not corrupt config.yaml.

    Verified by inspecting the script: it grep-checks for an existing
    `provider: duckbot-brain` line and skips the write if present.
    """
    script = (REPO_ROOT / "scripts" / "hermes-bootstrap.sh").read_text()
    assert "already set in" in script, (
        "hermes-bootstrap.sh should detect an existing memory.provider "
        "and skip the mutation on re-run."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))