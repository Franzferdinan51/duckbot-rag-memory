"""
test_openclaw_shim.py — verify the OpenClaw CLI shim.

Parallel to test_hermes_plugin.py but anchored at the shell entry point.
The shim delegates to the shared 9-tool surface in src.extensions.tools,
so this test mostly verifies the verb → dispatch mapping and that JSON
output is parseable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.connectors import openclaw_shim  # noqa: E402
from src.extensions import tools as _surface  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_state():
    _surface.reset_brain()
    from src import ratelimit
    ratelimit.reset_rate_limiter()
    yield
    _surface.reset_brain()
    ratelimit.reset_rate_limiter()


# -----------------------------------------------------------------------------
# main() dispatch
# -----------------------------------------------------------------------------


def test_main_no_args_prints_help_and_returns_zero(capsys):
    rc = openclaw_shim.main([])
    assert rc == 0
    out = capsys.readouterr().out
    # The docstring is the usage line; just verify something was printed.
    assert "openclaw" in out
    assert len(out) > 0


def test_main_help_flag_prints_help(capsys):
    rc = openclaw_shim.main(["--help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "python -m src.cli openclaw" in out


def test_main_unknown_verb_returns_error(capsys):
    rc = openclaw_shim.main(["does-not-exist"])
    assert rc == 1
    data = json.loads(capsys.readouterr().out)
    assert "error" in data
    assert "available" in data
    assert "wake-up" in data["available"]


# -----------------------------------------------------------------------------
# tools verb
# -----------------------------------------------------------------------------


def test_tools_verb_lists_eleven_tools(capsys):
    rc = openclaw_shim.main(["tools"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["tools"]) == 11
    assert data["tools"][0]["name"] == "brain_wake_up"
    assert "11 tools" in data["summary"]


# -----------------------------------------------------------------------------
# recall verb
# -----------------------------------------------------------------------------


def test_recall_requires_query():
    fake_brain = MagicMock()
    fake_brain.recall.return_value = []
    _surface._BRAIN = fake_brain
    rc = openclaw_shim.main(["recall"])
    assert rc == 2  # dispatch surfaced an error → exit 2


def test_recall_delegates_to_brain_with_query(capsys):
    fake_brain = MagicMock()
    fake_brain.recall.return_value = []
    _surface._BRAIN = fake_brain
    rc = openclaw_shim.main(["recall", "what", "about", "X", "-k", "3"])
    assert rc == 0
    fake_brain.recall.assert_called_once()
    args, kwargs = fake_brain.recall.call_args
    # The shared surface calls brain.recall(query=..., k=..., rerank=..., decay=...).
    assert kwargs["query"] == "what about X"
    assert kwargs["k"] == 3


def test_recall_default_k_is_five(capsys):
    fake_brain = MagicMock()
    fake_brain.recall.return_value = []
    _surface._BRAIN = fake_brain
    openclaw_shim.main(["recall", "some query"])
    args, kwargs = fake_brain.recall.call_args
    assert kwargs["k"] == 5


# -----------------------------------------------------------------------------
# remember verb
# -----------------------------------------------------------------------------


def test_remember_requires_text():
    rc = openclaw_shim.main(["remember"])
    assert rc == 2


def test_remember_returns_queued(capsys):
    fake_brain = MagicMock()
    _surface._BRAIN = fake_brain
    rc = openclaw_shim.main(["remember", "always", "run", "tests"])
    assert rc == 0
    data = json.loads(capsys.readouterr().out)
    assert data["status"] == "queued"
    assert data["source"] == "openclaw-cli://ad-hoc"


# -----------------------------------------------------------------------------
# wake-up verb
# -----------------------------------------------------------------------------


def test_wake_up_delegates(capsys):
    fake_brain = MagicMock()
    fake_brain.wake_up.return_value = {"memories": [], "blocks": []}
    _surface._BRAIN = fake_brain
    rc = openclaw_shim.main(["wake-up", "--query", "anchor", "-k", "4"])
    assert rc == 0
    fake_brain.wake_up.assert_called_once_with(
        query="anchor", k=4, include_blocks=True,
        include_graph=True, include_fsrs_review=True,
    )


def test_wake_up_queryless_uses_none(capsys):
    fake_brain = MagicMock()
    fake_brain.wake_up.return_value = {"memories": []}
    _surface._BRAIN = fake_brain
    openclaw_shim.main(["wake-up"])
    args, kwargs = fake_brain.wake_up.call_args
    if args:
        assert args[0] is None
    else:
        assert kwargs.get("query") is None


def test_wake_up_accepts_underscore_form():
    fake_brain = MagicMock()
    fake_brain.wake_up.return_value = {"memories": []}
    _surface._BRAIN = fake_brain
    rc = openclaw_shim.main(["wake_up"])
    assert rc == 0


# -----------------------------------------------------------------------------
# stats / fsrs-review / decay-status / search-verbatim
# -----------------------------------------------------------------------------


def test_stats_delegates(capsys):
    fake_brain = MagicMock()
    fake_brain.stats.return_value = MagicMock(
        vector_chunks=0, vector_by_tier={}, graph_entities=0,
        graph_relationships=0, graph_active_relationships=0,
        blocks=0, quarantine_total=0, quarantine_pending=0,
        quarantine_approved=0, quarantine_rejected=0, generated_at=0.0,
    )
    _surface._BRAIN = fake_brain
    rc = openclaw_shim.main(["stats"])
    assert rc == 0
    fake_brain.stats.assert_called_once_with()


def test_fsrs_review_with_tier_and_k(capsys):
    fake_brain = MagicMock()
    fake_brain.fsrs_review_queue.return_value = []
    _surface._BRAIN = fake_brain
    rc = openclaw_shim.main(["fsrs-review", "episodic", "-k", "7"])
    assert rc == 0
    fake_brain.fsrs_review_queue.assert_called_once_with(tier="episodic", k=7)


def test_decay_status_default(capsys):
    fake_brain = MagicMock()
    fake_brain.decay_status.return_value = {"tiers": {}}
    _surface._BRAIN = fake_brain
    rc = openclaw_shim.main(["decay-status"])
    assert rc == 0
    fake_brain.decay_status.assert_called_once_with(tier=None, k=50)


def test_search_verbatim_requires_needle():
    rc = openclaw_shim.main(["search-verbatim"])
    assert rc == 2


def test_search_verbatim_delegates(capsys):
    fake_brain = MagicMock()
    fake_brain.search_verbatim.return_value = []
    _surface._BRAIN = fake_brain
    rc = openclaw_shim.main(["search-verbatim", "literal", "phrase"])
    assert rc == 0
    fake_brain.search_verbatim.assert_called_once_with(needle="literal phrase", k=5)


# -----------------------------------------------------------------------------
# reflect verb
# -----------------------------------------------------------------------------


def test_reflect_default_lookback():
    """reflect with no args uses the default 7-day lookback."""
    # Memory().reflect is async + hits Chroma; patch the dispatch path.
    with patch.object(_surface, "dispatch", return_value={"promoted": 5}) as mock_disp:
        rc = openclaw_shim.main(["reflect"])
    assert rc == 0
    mock_disp.assert_called_once_with("brain_reflect", {})


def test_reflect_with_lookback():
    with patch.object(_surface, "dispatch", return_value={"promoted": 2}) as mock_disp:
        rc = openclaw_shim.main(["reflect", "14"])
    assert rc == 0
    mock_disp.assert_called_once_with("brain_reflect", {"lookback_days": 14})


# -----------------------------------------------------------------------------
# call verb (generic escape hatch)
# -----------------------------------------------------------------------------


def test_call_requires_tool_name():
    rc = openclaw_shim.main(["call"])
    assert rc == 2


def test_call_dispatches_with_json_args(capsys):
    with patch.object(_surface, "dispatch", return_value={"ok": True}) as mock_disp:
        rc = openclaw_shim.main(["call", "brain_recall", '{"query": "x", "k": 2}'])
    assert rc == 0
    mock_disp.assert_called_once_with("brain_recall", {"query": "x", "k": 2})


def test_call_no_args_passes_empty_dict(capsys):
    with patch.object(_surface, "dispatch", return_value={"ok": True}) as mock_disp:
        rc = openclaw_shim.main(["call", "brain_stats"])
    assert rc == 0
    mock_disp.assert_called_once_with("brain_stats", {})


def test_call_invalid_json_returns_error(capsys):
    rc = openclaw_shim.main(["call", "brain_recall", "{not json"])
    assert rc == 2
    data = json.loads(capsys.readouterr().out)
    assert "error" in data
    assert "JSON" in data["error"]


def test_call_non_object_json_returns_error(capsys):
    rc = openclaw_shim.main(["call", "brain_recall", "[1, 2, 3]"])
    assert rc == 2
    data = json.loads(capsys.readouterr().out)
    assert "error" in data


# -----------------------------------------------------------------------------
# exit codes
# -----------------------------------------------------------------------------


def test_dispatch_error_returns_exit_code_2(capsys):
    """When dispatch returns {"error": ...}, the shim exits 2 so shell
    pipelines can detect failure without parsing JSON."""
    fake_brain = MagicMock()
    fake_brain.recall.side_effect = RuntimeError("boom")
    _surface._BRAIN = fake_brain
    rc = openclaw_shim.main(["recall", "x"])
    assert rc == 2
