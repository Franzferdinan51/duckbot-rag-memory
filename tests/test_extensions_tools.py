"""
test_extensions_tools.py — verify the shared "core agent surface" used by
the OpenClaw extension adapter and the Hermes MemoryProvider plugin.

Pattern: import the tools module directly and exercise dispatch() against
a stub Brain so we don't hit Chroma during tests. Same as the per-entry-
point tests, but anchored at the SHARED layer (the layer the entry points
delegate to). This catches divergence early.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# Make `src` importable from the tests/ dir.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.extensions import tools as surface  # noqa: E402


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_shared_state():
    """Reset the Brain singleton + rate limiter between tests so the
    cross-test isolation matches the rest of the test suite."""
    surface.reset_brain()
    from src import ratelimit
    ratelimit.reset_rate_limiter()
    yield
    surface.reset_brain()
    ratelimit.reset_rate_limiter()


@pytest.fixture
def fake_brain():
    """A MagicMock that quacks like Brain enough for dispatch."""
    brain = MagicMock()
    brain.recall.return_value = []
    brain.recall_verbatim.return_value = []
    brain.stats.return_value = MagicMock(
        vector_chunks=10,
        vector_by_tier={"semantic": 6, "episodic": 4},
        graph_entities=2,
        graph_relationships=3,
        graph_active_relationships=3,
        blocks=1,
        quarantine_total=0,
        quarantine_pending=0,
        quarantine_approved=0,
        quarantine_rejected=0,
        generated_at=1234567890.0,
    )
    brain.fsrs_review_queue.return_value = []
    brain.decay_status.return_value = {"tiers": {}}
    brain.search_verbatim.return_value = []
    brain.wake_up.return_value = {
        "memories": [], "blocks": [],
        "graph_summary": {}, "fsrs_review_queue": [], "stats": {},
    }
    return brain


# -----------------------------------------------------------------------------
# TOOLS list shape
# -----------------------------------------------------------------------------


def test_tool_count_is_nine():
    """Surface is intentionally tight: 12 tools, the core agent subset."""
    assert len(surface.TOOLS) == 12


def test_tool_names_returns_eleven_in_canonical_order():
    names = surface.tool_names()
    assert len(names) == 12
    # brain_wake_up is the canonical first-call tool — it must be first.
    assert names[0] == "brain_wake_up"


def test_tools_have_required_mcp_fields():
    for t in surface.TOOLS:
        assert "name" in t, f"missing name: {t}"
        assert "description" in t, f"missing description: {t}"
        assert "inputSchema" in t, f"missing inputSchema: {t}"
        assert t["inputSchema"]["type"] == "object"


def test_tool_schemas_is_a_copy():
    """tool_schemas() must return a copy so callers can't mutate TOOLS."""
    schemas = surface.tool_schemas()
    schemas.append({"name": "x"})
    assert len(surface.TOOLS) == 12


def test_brain_wake_up_is_a_listed_tool():
    """The canonical session-start call must be on the surface."""
    names = {t["name"] for t in surface.TOOLS}
    assert "brain_wake_up" in names


def test_brain_recall_requires_query():
    recall = next(t for t in surface.TOOLS if t["name"] == "brain_recall")
    assert "query" in recall["inputSchema"]["required"]


def test_brain_remember_requires_text():
    rem = next(t for t in surface.TOOLS if t["name"] == "brain_remember")
    assert "text" in rem["inputSchema"]["required"]


def test_brain_search_verbatim_requires_needle():
    sv = next(t for t in surface.TOOLS if t["name"] == "brain_search_verbatim")
    assert "needle" in sv["inputSchema"]["required"]


# -----------------------------------------------------------------------------
# function-call-shape (Hermes plugin consumes this)
# -----------------------------------------------------------------------------


def test_function_call_schemas_match_tool_count():
    schemas = surface.function_call_schemas()
    assert len(schemas) == 12
    for s in schemas:
        assert s["type"] == "function"
        assert "name" in s["function"]
        assert "description" in s["function"]
        assert "parameters" in s["function"]


# -----------------------------------------------------------------------------
# system_prompt_block
# -----------------------------------------------------------------------------


def test_system_prompt_block_mentions_wake_up_first():
    """Skill files say 'call brain_wake_up first on session start' — the
    prompt block must reflect that."""
    block = surface.system_prompt_block()
    wake_idx = block.find("brain_wake_up")
    recall_idx = block.find("brain_recall")
    assert wake_idx > 0
    assert recall_idx > wake_idx, "brain_wake_up must be listed before brain_recall"


# -----------------------------------------------------------------------------
# dispatch
# -----------------------------------------------------------------------------


def test_dispatch_unknown_tool_returns_error_dict():
    out = surface.dispatch("brain_does_not_exist", {})
    assert "error" in out
    assert "brain_does_not_exist" in out["error"]


def test_dispatch_brain_recall_delegates(fake_brain):
    surface._BRAIN = fake_brain
    chunk = MagicMock(
        chunk_id="c1", text="t", tier="semantic",
        importance=0.5, score=0.1, source_path="/x.md", metadata={},
    )
    fake_brain.recall.return_value = [chunk]
    out = surface.dispatch("brain_recall", {"query": "q", "k": 3})
    assert "results" in out
    assert out["results"][0]["chunk_id"] == "c1"
    fake_brain.recall.assert_called_once_with(
        query="q", k=3, tier=None, min_importance=None,
        rerank=False, decay=False,
        tier_priors=False, tier_priors_overrides=None, fsrs=False,
    )


def test_dispatch_brain_recall_verbatim_delegates(fake_brain):
    surface._BRAIN = fake_brain
    fake_brain.recall_verbatim.return_value = [{"verbatim_text": "src"}]
    out = surface.dispatch("brain_recall_verbatim", {"query": "q"})
    assert "results" in out
    assert out["results"][0]["verbatim_text"] == "src"


def test_dispatch_brain_stats_serializes_dataclass(fake_brain):
    surface._BRAIN = fake_brain
    out = surface.dispatch("brain_stats", {})
    assert out["vector_chunks"] == 10
    assert out["vector_by_tier"]["semantic"] == 6
    assert out["generated_at"] == 1234567890.0


def test_dispatch_brain_search_verbatim_delegates(fake_brain):
    surface._BRAIN = fake_brain
    fake_brain.search_verbatim.return_value = [{"chunk_id": "x"}]
    out = surface.dispatch("brain_search_verbatim", {"needle": "abc"})
    assert "matches" in out
    fake_brain.search_verbatim.assert_called_once_with(needle="abc", k=5)


def test_dispatch_brain_fsrs_review_delegates(fake_brain):
    surface._BRAIN = fake_brain
    fake_brain.fsrs_review_queue.return_value = [{"chunk_id": "due"}]
    out = surface.dispatch("brain_fsrs_review", {"tier": "episodic", "k": 5})
    assert "queue" in out
    fake_brain.fsrs_review_queue.assert_called_once_with(tier="episodic", k=5)


def test_dispatch_brain_decay_status_delegates(fake_brain):
    surface._BRAIN = fake_brain
    out = surface.dispatch("brain_decay_status", {"k": 25})
    assert "tiers" in out
    fake_brain.decay_status.assert_called_once_with(tier=None, k=25)


def test_dispatch_brain_wake_up_delegates(fake_brain):
    surface._BRAIN = fake_brain
    out = surface.dispatch("brain_wake_up", {"k": 4, "query": "anchor"})
    assert "memories" in out
    fake_brain.wake_up.assert_called_once_with(
        query="anchor", k=4, include_blocks=True,
        include_graph=True, include_fsrs_review=True,
    )


def test_dispatch_brain_wake_up_queryless(fake_brain):
    """Empty-string query → passed as None so Brain.wake_up uses the
    recent-memory path, not the recall path."""
    surface._BRAIN = fake_brain
    surface.dispatch("brain_wake_up", {"query": ""})
    args, kwargs = fake_brain.wake_up.call_args
    # query is the first positional OR a keyword; check both shapes.
    if args:
        assert args[0] is None
    else:
        assert kwargs.get("query") is None


def test_brain_wake_up_description_matches_queryless_behavior():
    wake = next(t for t in surface.TOOLS if t["name"] == "brain_wake_up")
    desc = wake["description"]
    assert "blank query" in desc.lower()
    assert "query=''" not in desc


def test_dispatch_brain_remember_is_non_blocking(fake_brain):
    """brain_remember returns immediately with status=queued, even
    before the background thread runs."""
    surface._BRAIN = fake_brain
    out = surface.dispatch("brain_remember", {"text": "remember this", "source": "test://"})
    assert out["status"] == "queued"
    assert out["source"] == "test://"


def test_dispatch_handles_missing_required_arg(fake_brain):
    """Missing required arg → error dict, not exception."""
    surface._BRAIN = fake_brain
    out = surface.dispatch("brain_recall", {})  # no query
    assert "error" in out
    assert "query" in out["error"]


def test_dispatch_handles_brain_exception(fake_brain):
    """If the Brain raises, dispatch returns an error dict (no exception
    escapes — entry points need a stable envelope)."""
    surface._BRAIN = fake_brain
    fake_brain.recall.side_effect = RuntimeError("chroma down")
    out = surface.dispatch("brain_recall", {"query": "q"})
    assert "error" in out
    assert "chroma down" in out["error"]


# -----------------------------------------------------------------------------
# rate-limit guard
# -----------------------------------------------------------------------------


def test_check_rate_limit_returns_none_when_allowed():
    out = surface.check_rate_limit("brain_recall")
    assert out is None


def test_check_rate_limit_returns_error_when_exhausted():
    """Burn the bucket so the next call returns a rate_limited error."""
    from src import ratelimit
    rl = ratelimit.get_rate_limiter()
    # brain_remember has limit=10/min — burn it past exhaustion.
    for _ in range(50):
        rl.check("brain_remember")
    out = surface.check_rate_limit("brain_remember")
    assert out is not None
    assert out["error"] == "rate_limited"
    assert out["tool"] == "brain_remember"
    assert out["limit_per_min"] == 10
    assert out["retry_after_seconds"] > 0
    assert "DUCKBOT_RATELIMIT_DISABLE" in out["message"]


def test_check_rate_limit_disabled_env():
    """DUCKBOT_RATELIMIT_DISABLE=1 turns the check off."""
    import os
    os.environ["DUCKBOT_RATELIMIT_DISABLE"] = "1"
    try:
        for _ in range(100):
            assert surface.check_rate_limit("brain_remember") is None
    finally:
        del os.environ["DUCKBOT_RATELIMIT_DISABLE"]


# -----------------------------------------------------------------------------
# summary() — sanity-check the one-liner used by `python -c "from src..."`
# -----------------------------------------------------------------------------


def test_summary_mentions_all_twelve_tools():
    out = surface.summary()
    for name in surface.tool_names():
        assert name in out
    assert "12 tools" in out

def test_dispatch_brain_search_verbatim_rejects_empty_needle(fake_brain):
    """Empty needle MUST be rejected — `in ""` matches every chunk,
    returning expensive semantic results for no useful query."""
    surface._BRAIN = fake_brain
    out = surface.dispatch("brain_search_verbatim", {"needle": ""})
    assert "error" in out
    assert "non-empty" in out["error"]
    fake_brain.search_verbatim.assert_not_called()


def test_dispatch_brain_search_verbatim_rejects_whitespace_needle(fake_brain):
    """Whitespace-only needle is equivalent to empty — same rejection."""
    surface._BRAIN = fake_brain
    out = surface.dispatch("brain_search_verbatim", {"needle": "   "})
    assert "error" in out
    fake_brain.search_verbatim.assert_not_called()


def test_dispatch_brain_search_verbatim_strips_whitespace(fake_brain):
    """Needle is stripped before dispatch so leading/trailing whitespace
    doesn't miss the match."""
    fake_brain.search_verbatim.return_value = [{"chunk_id": "x"}]
    surface._BRAIN = fake_brain
    out = surface.dispatch("brain_search_verbatim", {"needle": "  docker compose  "})
    assert "matches" in out
    fake_brain.search_verbatim.assert_called_once_with(needle="docker compose", k=5)


def test_dispatch_brain_remember_rejects_empty_text(fake_brain):
    """Empty text MUST be rejected — otherwise the daemon thread silently
    fails (background Chroma error) and the agent thinks it succeeded."""
    surface._BRAIN = fake_brain
    out = surface.dispatch("brain_remember", {"text": ""})
    assert "error" in out
    assert "non-empty" in out["error"]


def test_dispatch_brain_remember_rejects_whitespace_text(fake_brain):
    surface._BRAIN = fake_brain
    out = surface.dispatch("brain_remember", {"text": "   \t\n  "})
    assert "error" in out
    assert "non-empty" in out["error"]


def test_dispatch_brain_remember_rejects_empty_skill_candidate(fake_brain):
    """Empty skill_candidate would otherwise stamp a chunk_id = sha256("")
    + source — an empty useless chunk in the procedural tier forever."""
    surface._BRAIN = fake_brain
    fake_brain.remember.return_value = MagicMock(chunk_id="abc", tier="procedural", stored=True)
    out = surface.dispatch("brain_remember", {"text": "", "kind": "skill_candidate"})
    assert "error" in out
    # stamp_skill_candidate should NOT have been called
    fake_brain.remember.assert_not_called()


def test_dispatch_brain_recall_passes_tier_priors_and_fsrs(fake_brain):
    """tier_priors / tier_priors_overrides / fsrs must reach Brain.recall().
    Previously these params were in Brain.recall() but the dispatch
    silently dropped them — agents calling brain_recall(tier_priors=true)
    got default recall behavior with no error."""
    fake_brain.recall.return_value = []
    surface._BRAIN = fake_brain
    out = surface.dispatch("brain_recall", {
        "query": "q",
        "tier_priors": True,
        "tier_priors_overrides": {"procedural": 2.0},
        "fsrs": True,
    })
    fake_brain.recall.assert_called_once_with(
        query="q", k=5, tier=None, min_importance=None,
        rerank=False, decay=False,
        tier_priors=True, tier_priors_overrides={"procedural": 2.0}, fsrs=True,
    )


def test_dispatch_brain_recall_rejects_non_dict_overrides(fake_brain):
    """tier_priors_overrides must be a dict (or absent) — other types
    would crash inside Brain.recall() with a confusing error."""
    surface._BRAIN = fake_brain
    out = surface.dispatch("brain_recall", {
        "query": "q",
        "tier_priors_overrides": "not a dict",
    })
    assert "error" in out
    assert "must be a dict" in out["error"]
    fake_brain.recall.assert_not_called()
