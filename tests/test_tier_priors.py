"""
test_tier_priors.py — verify the per-tier RRF prior weighting (Layer 11).

duckbot-secret-scan: allowlist-file
"""

# duckbot-secret-scan: allowlist-file
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest


ROOT = "/Users/duckets/Desktop/duckbot-rag-memory"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.tier_priors import (  # noqa: E402
    DEFAULT_PRIORS,
    PRIOR_MAX,
    PRIOR_MIN,
    get_prior,
    maybe_apply_tier_priors,
)


# Minimal stand-in for QueryResult.
@dataclass
class FakeResult:
    chunk_id: str
    text: str
    tier: str
    rrf_score: float
    metadata: dict | None = None


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------


def test_default_priors_match_design():
    """Procedural > semantic > episodic > working, in this order."""
    assert DEFAULT_PRIORS["procedural"] > DEFAULT_PRIORS["semantic"]
    assert DEFAULT_PRIORS["semantic"] > DEFAULT_PRIORS["episodic"]
    assert DEFAULT_PRIORS["episodic"] > DEFAULT_PRIORS["working"]
    # Exact values from the L11 design.
    assert DEFAULT_PRIORS["procedural"] == 1.50
    assert DEFAULT_PRIORS["semantic"] == 1.20
    assert DEFAULT_PRIORS["episodic"] == 1.00
    assert DEFAULT_PRIORS["working"] == 0.80


def test_get_prior_unknown_tier_returns_baseline():
    """Unknown tier names get 1.0 (episodic baseline)."""
    assert get_prior("nonsense") == 1.0


def test_get_prior_clamp_low():
    """Priors below PRIOR_MIN get clamped up."""
    p = get_prior("procedural", overrides={"procedural": 0.001})
    assert p == PRIOR_MIN


def test_get_prior_clamp_high():
    """Priors above PRIOR_MAX get clamped down."""
    p = get_prior("procedural", overrides={"procedural": 9999})
    assert p == PRIOR_MAX


def test_get_prior_override_takes_precedence():
    p = get_prior("procedural", overrides={"procedural": 2.0})
    assert p == 2.0


# -----------------------------------------------------------------------------
# Opt-in behavior
# -----------------------------------------------------------------------------


def test_disabled_returns_input_unchanged():
    """If enabled=False, the input list is returned untouched (no re-sort, no mutation)."""
    r1 = FakeResult("a", "alpha", "procedural", 0.5)
    r2 = FakeResult("b", "beta", "working", 0.9)  # higher base RRF
    out = maybe_apply_tier_priors([r1, r2], enabled=False)
    # Disabled: NO re-sort, NO mutation. Input order preserved.
    assert [r.chunk_id for r in out] == ["a", "b"]
    # Scores must not be touched.
    assert out[0].rrf_score == 0.5
    assert out[1].rrf_score == 0.9


def test_disabled_via_env(monkeypatch):
    """No env var set + enabled=None → no-op (DUCKBOT_TIER_PRIORS unset)."""
    monkeypatch.delenv("DUCKBOT_TIER_PRIORS", raising=False)
    r1 = FakeResult("a", "alpha", "procedural", 0.5)
    r2 = FakeResult("b", "beta", "working", 0.9)
    out = maybe_apply_tier_priors([r1, r2], enabled=None)
    # Disabled: input order preserved (no re-sort).
    assert [r.chunk_id for r in out] == ["a", "b"]


def test_enabled_via_env(monkeypatch):
    """DUCKBOT_TIER_PRIORS=1 enables priors without explicit kwarg."""
    monkeypatch.setenv("DUCKBOT_TIER_PRIORS", "1")
    r1 = FakeResult("a", "alpha", "procedural", 0.5)  # 0.5 * 1.5 = 0.75
    r2 = FakeResult("b", "beta", "working", 0.9)      # 0.9 * 0.8 = 0.72
    out = maybe_apply_tier_priors([r1, r2], enabled=None)
    # With priors: procedural 0.75 > working 0.72 → procedural wins.
    assert out[0].chunk_id == "a"
    assert out[1].chunk_id == "b"


def test_enabled_via_kwarg(monkeypatch):
    """Explicit enabled=True wins over the env var (which is unset)."""
    monkeypatch.delenv("DUCKBOT_TIER_PRIORS", raising=False)
    r1 = FakeResult("a", "alpha", "procedural", 0.5)
    r2 = FakeResult("b", "beta", "working", 0.9)
    out = maybe_apply_tier_priors([r1, r2], enabled=True)
    assert out[0].chunk_id == "a"


def test_explicit_false_wins_over_env(monkeypatch):
    """Explicit enabled=False forces off even if DUCKBOT_TIER_PRIORS=1."""
    monkeypatch.setenv("DUCKBOT_TIER_PRIORS", "1")
    r1 = FakeResult("a", "alpha", "procedural", 0.5)
    r2 = FakeResult("b", "beta", "working", 0.9)
    out = maybe_apply_tier_priors([r1, r2], enabled=False)
    # Disabled: input order preserved.
    assert [r.chunk_id for r in out] == ["a", "b"]


def test_empty_input_returns_empty():
    assert maybe_apply_tier_priors([], enabled=True) == []


# -----------------------------------------------------------------------------
# Math correctness
# -----------------------------------------------------------------------------


def test_procedural_boost_1_5x():
    """Procedural prior multiplies RRF by 1.5."""
    r = FakeResult("a", "alpha", "procedural", 1.0)
    [out] = maybe_apply_tier_priors([r], enabled=True)
    assert abs(out.rrf_score - 1.5) < 1e-9


def test_working_demote_0_8x():
    """Working prior multiplies RRF by 0.8."""
    r = FakeResult("a", "alpha", "working", 1.0)
    [out] = maybe_apply_tier_priors([r], enabled=True)
    assert abs(out.rrf_score - 0.8) < 1e-9


def test_episodic_baseline_unchanged():
    """Episodic prior is 1.0 (no change to score)."""
    r = FakeResult("a", "alpha", "episodic", 0.42)
    [out] = maybe_apply_tier_priors([r], enabled=True)
    assert abs(out.rrf_score - 0.42) < 1e-9


def test_semantic_boost_1_2x():
    """Semantic prior multiplies RRF by 1.2."""
    r = FakeResult("a", "alpha", "semantic", 1.0)
    [out] = maybe_apply_tier_priors([r], enabled=True)
    assert abs(out.rrf_score - 1.2) < 1e-9


def test_audit_field_attached():
    """_tier_prior and _rrf_score_pre_prior are set on each result."""
    r = FakeResult("a", "alpha", "procedural", 0.5)
    [out] = maybe_apply_tier_priors([r], enabled=True)
    assert out._tier_prior == 1.5
    assert abs(out._rrf_score_pre_prior - 0.5) < 1e-9


def test_results_sorted_descending_by_adjusted_score():
    """After applying priors, results are sorted by adjusted score desc."""
    # All have the same RRF=1.0; their tier priors decide order.
    items = [
        FakeResult("work", "w", "working", 1.0),     # 0.8
        FakeResult("proc", "p", "procedural", 1.0),  # 1.5
        FakeResult("ep", "e", "episodic", 1.0),      # 1.0
        FakeResult("sem", "s", "semantic", 1.0),     # 1.2
    ]
    out = maybe_apply_tier_priors(items, enabled=True)
    assert [r.chunk_id for r in out] == ["proc", "sem", "ep", "work"]


def test_override_prior_applied():
    """Custom prior override beats the default."""
    r = FakeResult("a", "alpha", "procedural", 1.0)
    [out] = maybe_apply_tier_priors([r], enabled=True, overrides={"procedural": 3.0})
    assert out.rrf_score == 3.0


def test_override_for_other_tier_uses_default():
    """Override for one tier doesn't affect other tiers' defaults."""
    r1 = FakeResult("a", "alpha", "procedural", 1.0)  # overridden to 3.0
    r2 = FakeResult("b", "beta", "working", 1.0)      # default 0.8
    out = maybe_apply_tier_priors([r1, r2], enabled=True, overrides={"procedural": 3.0})
    by_id = {r.chunk_id: r for r in out}
    assert by_id["a"].rrf_score == 3.0
    assert by_id["b"].rrf_score == 0.8


def test_unknown_tier_falls_back_to_1():
    """Chunks with an unknown tier name get a 1.0 prior (episodic baseline)."""
    r = FakeResult("a", "alpha", "mystery", 0.5)
    [out] = maybe_apply_tier_priors([r], enabled=True)
    assert out.rrf_score == 0.5  # unchanged because prior=1.0


# -----------------------------------------------------------------------------
# Real QueryResult (from src/query.py) round-trip
# -----------------------------------------------------------------------------


def test_real_query_result_round_trip():
    """Apply priors to actual QueryResult objects (with metadata dict)."""
    from src.query import QueryResult
    items = [
        QueryResult(chunk_id="a", text="alpha", metadata={}, tier="procedural", rrf_score=1.0),
        QueryResult(chunk_id="b", text="beta", metadata={}, tier="working", rrf_score=1.0),
    ]
    out = maybe_apply_tier_priors(items, enabled=True)
    assert out[0].chunk_id == "a"  # procedural boosted
    assert out[1].chunk_id == "b"
    assert abs(out[0].rrf_score - 1.5) < 1e-9
    assert abs(out[1].rrf_score - 0.8) < 1e-9
    # Audit fields set.
    assert out[0]._tier_prior == 1.5
    assert out[1]._tier_prior == 0.8
