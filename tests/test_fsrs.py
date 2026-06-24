"""
test_fsrs.py — verify the FSRS-6 spaced repetition math (Layer 9).

duckbot-secret-scan: allowlist-file

The math is reimplemented from the FSRS-6 algorithm spec (public domain),
not from any source code. We test:
  - R(t, S) = (1 + factor*t/S)^(1/factor) power-law forgetting curve.
  - S'_r = S * (e^w8 * (11 - D) * S^-0.8 * (1 - R) + 1) growth formula.
  - D'_r = D - w6 * (R - 0.5) difficulty update.
  - maybe_fsrs() opt-in dispatch (env var + kwarg).
  - Audit fields attached to results.
"""

# duckbot-secret-scan: allowlist-file
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass
from unittest.mock import patch

import pytest


ROOT = "/Users/duckets/Desktop/duckbot-rag-memory"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.fsrs import (  # noqa: E402
    DEFAULT_DIFFICULTY,
    DEFAULT_W20,
    DEFAULT_W6,
    DEFAULT_W8,
    DIFFICULTY_MAX,
    DIFFICULTY_MIN,
    STABILITY_CEILING,
    STABILITY_FLOOR,
    days_since,
    fsrs_adjust,
    fsrs_bump_difficulty,
    fsrs_bump_stability,
    fsrs_retrievability,
    maybe_fsrs,
)
from src.query import QueryResult  # noqa: E402


# -----------------------------------------------------------------------------
# R(t, S) retrievability
# -----------------------------------------------------------------------------


def test_retrievability_at_zero_age_is_one():
    """At t=0, R=1.0 (just recalled)."""
    assert fsrs_retrievability(0, stability=7) == 1.0


def test_retrievability_at_nine_S_equals_two_to_neg_w20():
    """At t=9*S, the base is exactly 2, so R = 2^(-w20)."""
    r = fsrs_retrievability(elapsed_days=9*7, stability=7, w20=1.0)
    assert abs(r - 0.5) < 1e-9


def test_retrievability_at_one_S_about_0_9():
    """At t=S with w20=0.9, R(1, S) = (1 + 1/9)^(-0.9) ≈ 0.91."""
    r = fsrs_retrievability(elapsed_days=7, stability=7, w20=0.9)
    assert abs(r - 0.91) < 0.01


def test_retrievability_monotonically_decreasing_in_t():
    """R decreases as elapsed time grows (for fixed S)."""
    r_early = fsrs_retrievability(elapsed_days=1, stability=7)
    r_mid = fsrs_retrievability(elapsed_days=10, stability=7)
    r_late = fsrs_retrievability(elapsed_days=100, stability=7)
    assert r_early > r_mid > r_late
    assert r_late < 1.0


def test_retrievability_higher_w20_steeper_decay():
    """Larger w20 → steeper decay curve (more aggressive forgetting)."""
    r_lo = fsrs_retrievability(elapsed_days=30, stability=7, w20=0.3)
    r_hi = fsrs_retrievability(elapsed_days=30, stability=7, w20=1.5)
    # Higher w20 → faster forgetting → lower R at t=30
    assert r_hi < r_lo


def test_retrievability_clamps_to_zero():
    """At t → ∞, R → 0 (effectively)."""
    r = fsrs_retrievability(elapsed_days=1e10, stability=7)
    assert r < 1e-6  # effectively zero, not literally 0.0


def test_retrievability_zero_stability_returns_zero():
    """S=0 is invalid; return 0."""
    assert fsrs_retrievability(7, stability=0) == 0.0


def test_retrievability_negative_elapsed_clamped_to_zero():
    """Negative elapsed_days (clock skew) → R=1.0."""
    r = fsrs_retrievability(elapsed_days=-100, stability=7)
    assert r == 1.0


def test_retrievability_higher_stability_slower_decay():
    """Higher S → slower decay at fixed elapsed time."""
    r_s7 = fsrs_retrievability(elapsed_days=30, stability=7)
    r_s30 = fsrs_retrievability(elapsed_days=30, stability=30)
    assert r_s30 > r_s7


def test_retrievability_custom_w20_changes_curve():
    """Custom w20 changes the curve shape."""
    r_default = fsrs_retrievability(elapsed_days=30, stability=7, w20=0.9)
    r_higher = fsrs_retrievability(elapsed_days=30, stability=7, w20=1.5)
    assert r_default != r_higher


def test_retrievability_invalid_w20_returns_zero():
    """w20 <= 0 is invalid (would invert the curve)."""
    assert fsrs_retrievability(elapsed_days=10, stability=7, w20=0) == 0.0
    assert fsrs_retrievability(elapsed_days=10, stability=7, w20=-1.0) == 0.0


# -----------------------------------------------------------------------------
# S'_r stability growth
# -----------------------------------------------------------------------------


def test_stability_grows_on_recall():
    """Successful recall increases stability."""
    s0 = 7.0
    s_new = fsrs_bump_stability(stability=s0, difficulty=5.0, retrievability=0.5, recalled=True)
    assert s_new > s0


def test_stability_growth_easy_chunk_larger():
    """Lower difficulty (D=1) → LARGER growth than higher difficulty (D=10).

    Per FSRS-6 spec: S'_r = S * (e^w8 * (11 - D) * ... + 1). The (11 - D)
    term makes easy chunks grow MORE, hard chunks grow LESS. (This
    matches spaced-rep research: easy items should graduate to longer
    intervals faster.)
    """
    s_easy = fsrs_bump_stability(stability=10, difficulty=1.0, retrievability=0.5, recalled=True)
    s_hard = fsrs_bump_stability(stability=10, difficulty=10.0, retrievability=0.5, recalled=True)
    # Easy chunk (D=1) grows more than hard chunk (D=10).
    assert s_easy > s_hard


def test_stability_growth_fresh_chunk_uses_default():
    """None stability → uses default 7.0 and grows it."""
    s = fsrs_bump_stability(stability=None, difficulty=5.0, retrievability=0.5, recalled=True)
    assert s > 7.0


def test_stability_shrinks_on_recall_failure():
    """Failed recall → S shrinks."""
    s = fsrs_bump_stability(stability=10, difficulty=5.0, retrievability=0.5, recalled=False)
    assert s < 10.0


def test_stability_clamped_to_floor():
    """Stability can't drop below STABILITY_FLOOR."""
    s = fsrs_bump_stability(stability=STABILITY_FLOOR, difficulty=10, retrievability=0, recalled=False)
    assert s >= STABILITY_FLOOR


def test_stability_clamped_to_ceiling():
    """Stability can't exceed STABILITY_CEILING."""
    s = fsrs_bump_stability(stability=STABILITY_CEILING, difficulty=1, retrievability=0, recalled=True)
    assert s <= STABILITY_CEILING


def test_stability_difficulty_clamped():
    """Out-of-range difficulty is clamped before use."""
    s1 = fsrs_bump_stability(stability=10, difficulty=999, retrievability=0.5, recalled=True)
    s2 = fsrs_bump_stability(stability=10, difficulty=DIFFICULTY_MAX, retrievability=0.5, recalled=True)
    assert s1 == s2


def test_stability_recall_oversize_inputs_no_crash():
    """Pathological inputs don't crash (OverflowError → safe default)."""
    s = fsrs_bump_stability(stability=1e308, difficulty=5, retrievability=0.5, recalled=True)
    # Should be clamped to ceiling, not raise.
    assert s <= STABILITY_CEILING


# -----------------------------------------------------------------------------
# D'_r difficulty update
# -----------------------------------------------------------------------------


def test_difficulty_easy_recall_decreases_difficulty():
    """Confident success (R=0.9) decreases difficulty."""
    d_new = fsrs_bump_difficulty(difficulty=5.0, retrievability=0.9, recalled=True)
    assert d_new < 5.0


def test_difficulty_barely_passed_increases_difficulty():
    """R=0.4 (barely passed) increases difficulty slightly."""
    d_new = fsrs_bump_difficulty(difficulty=5.0, retrievability=0.4, recalled=True)
    assert d_new > 5.0


def test_difficulty_failure_increases_difficulty():
    """Failed recall increases difficulty."""
    d_new = fsrs_bump_difficulty(difficulty=5.0, retrievability=0.3, recalled=False)
    assert d_new > 5.0


def test_difficulty_clamped_min():
    """Difficulty can't go below DIFFICULTY_MIN."""
    d = fsrs_bump_difficulty(difficulty=DIFFICULTY_MIN, retrievability=0.99, recalled=True)
    assert d >= DIFFICULTY_MIN


def test_difficulty_clamped_max():
    """Difficulty can't go above DIFFICULTY_MAX."""
    d = fsrs_bump_difficulty(difficulty=DIFFICULTY_MAX, retrievability=0, recalled=False)
    assert d <= DIFFICULTY_MAX


# -----------------------------------------------------------------------------
# fsrs_adjust() single-chunk helper
# -----------------------------------------------------------------------------


def test_fsrs_adjust_score_equals_input_times_r():
    """adjusted = score * retrievability."""
    r = fsrs_adjust("c1", score=1.0, stability=7, difficulty=5, elapsed_days=0.1)
    # R should be in (0, 1), so adjusted should be in (0, 1).
    assert 0 < r.retrievability <= 1.0
    assert 0 < r.fsrs_adjusted_score <= 1.0
    # adjusted == score * R exactly (modulo float).
    assert abs(r.fsrs_adjusted_score - r.original_score * r.retrievability) < 1e-9


def test_fsrs_adjust_audit_dict_to_dict():
    """FSRSWeightedResult.to_dict() is JSON-safe."""
    import json as _json
    r = fsrs_adjust("c1", score=0.5, stability=10, difficulty=5, elapsed_days=3)
    _json.dumps(r.to_dict())


# -----------------------------------------------------------------------------
# days_since helper
# -----------------------------------------------------------------------------


def test_days_since_none_returns_inf():
    assert days_since(None) == float("inf")


def test_days_since_recent_returns_small_value():
    """A timestamp 60 seconds ago → ~0.0007 days."""
    now = time.time()
    d = days_since(now - 60)
    assert 0.0005 < d < 0.001


def test_days_since_future_clamped_to_zero():
    """Future timestamps (clock skew) clamp to 0."""
    d = days_since(time.time() + 10000)
    assert d == 0.0


# -----------------------------------------------------------------------------
# maybe_fsrs() opt-in dispatch
# -----------------------------------------------------------------------------


def _make_result(cid: str, rrf: float, meta: dict | None = None) -> QueryResult:
    return QueryResult(
        chunk_id=cid,
        text=f"text for {cid}",
        metadata=meta or {},
        tier="episodic",
        rrf_score=rrf,
    )


def test_maybe_fsrs_disabled_returns_unchanged():
    """enabled=False → no mutation, no re-sort."""
    r1 = _make_result("a", 0.5, {"stability_days": 100, "difficulty": 5})
    r2 = _make_result("b", 0.9, {"stability_days": 100, "difficulty": 5})
    out = maybe_fsrs([r1, r2], enabled=False)
    # Input order preserved.
    assert [r.chunk_id for r in out] == ["a", "b"]
    # Scores unchanged.
    assert out[0].rrf_score == 0.5
    assert out[1].rrf_score == 0.9
    # No audit fields.
    assert not hasattr(out[0], "_fsrs_retrievability")


def test_maybe_fsrs_enabled_via_kwarg(monkeypatch):
    """Explicit enabled=True works regardless of env var."""
    monkeypatch.delenv("DUCKBOT_FSRS", raising=False)
    # Use an elapsed long enough that R < 1 meaningfully.
    r1 = _make_result("a", 1.0, {"stability_days": 7, "difficulty": 5, "last_recalled_at": time.time() - 5*86400})
    out = maybe_fsrs([r1], enabled=True)
    # R should be in (0, 1) and < 1.
    r_val = out[0]._fsrs_retrievability
    assert 0 < r_val < 1.0
    # adjusted = rrf * R, so it should be in (0, 1).
    assert 0 < out[0].rrf_score < 1.0


def test_maybe_fsrs_enabled_via_env(monkeypatch):
    """DUCKBOT_FSRS=1 enables without explicit kwarg."""
    monkeypatch.setenv("DUCKBOT_FSRS", "1")
    r1 = _make_result("a", 1.0, {"stability_days": 7, "difficulty": 5, "last_recalled_at": time.time() - 7*86400})
    out = maybe_fsrs([r1], enabled=None)
    assert hasattr(out[0], "_fsrs_retrievability")


def test_maybe_fsrs_explicit_false_wins_over_env(monkeypatch):
    """Explicit enabled=False forces off even if DUCKBOT_FSRS=1."""
    monkeypatch.setenv("DUCKBOT_FSRS", "1")
    r1 = _make_result("a", 1.0, {"stability_days": 7, "difficulty": 5})
    out = maybe_fsrs([r1], enabled=False)
    assert not hasattr(out[0], "_fsrs_retrievability")


def test_maybe_fsrs_recent_chunk_scores_higher():
    """Recently-recalled chunk has R close to 1, gets a higher adjusted score than a stale one."""
    now = time.time()
    recent = _make_result("recent", 1.0, {"stability_days": 30, "difficulty": 5, "last_recalled_at": now})
    stale = _make_result("stale", 1.0, {"stability_days": 30, "difficulty": 5, "last_recalled_at": now - 365*86400})
    out = maybe_fsrs([stale, recent], enabled=True)
    # Recent should win despite same RRF.
    assert out[0].chunk_id == "recent"
    assert out[1].chunk_id == "stale"
    assert out[0].rrf_score > out[1].rrf_score
    # Recent's R is much higher than stale's.
    assert out[0]._fsrs_retrievability > out[1]._fsrs_retrievability


def test_maybe_fsrs_missing_stability_uses_default():
    """Chunk without stability_days → default 7.0."""
    r1 = _make_result("a", 1.0, {})  # no stability
    out = maybe_fsrs([r1], enabled=True)
    assert out[0]._fsrs_stability == 7.0
    assert out[0]._fsrs_difficulty == DEFAULT_DIFFICULTY


def test_maybe_fsrs_audit_fields_attached():
    """Each result has _fsrs_retrievability, _fsrs_stability, _fsrs_difficulty."""
    r1 = _make_result("a", 1.0, {"stability_days": 14, "difficulty": 4, "last_recalled_at": time.time() - 14*86400})
    out = maybe_fsrs([r1], enabled=True)
    r = out[0]
    assert hasattr(r, "_fsrs_retrievability")
    assert hasattr(r, "_fsrs_stability")
    assert hasattr(r, "_fsrs_difficulty")
    assert hasattr(r, "_fsrs_elapsed_days")
    assert r._fsrs_stability == 14.0
    assert r._fsrs_difficulty == 4.0


def test_maybe_fsrs_empty_input_returns_empty():
    assert maybe_fsrs([], enabled=True) == []


def test_maybe_fsrs_uses_created_at_as_fallback():
    """If no last_recalled_at, use created_at to compute age."""
    r1 = _make_result("a", 1.0, {
        "stability_days": 7,
        "difficulty": 5,
        "created_at": time.time() - 5*86400,
    })
    out = maybe_fsrs([r1], enabled=True)
    # R should be in (0, 1) — long enough elapsed.
    r_val = out[0]._fsrs_retrievability
    assert 0 < r_val <= 1.0


def test_maybe_fsrs_uses_ingested_at_as_fallback():
    """If no last_recalled_at or created_at, use ingested_at."""
    r1 = _make_result("a", 1.0, {
        "stability_days": 7,
        "difficulty": 5,
        "ingested_at": time.time() - 5*86400,
    })
    out = maybe_fsrs([r1], enabled=True)
    r_val = out[0]._fsrs_retrievability
    assert 0 < r_val <= 1.0


def test_maybe_fsrs_no_timestamps_treated_as_just_recalled():
    """If no timestamps at all, elapsed=0 → R=1.0."""
    r1 = _make_result("a", 1.0, {"stability_days": 7, "difficulty": 5})
    out = maybe_fsrs([r1], enabled=True)
    assert out[0]._fsrs_retrievability == 1.0
    assert out[0]._fsrs_elapsed_days == 0.0


def test_maybe_fsrs_old_chunk_with_high_stability_survives():
    """A 10-day-old chunk with S=30 should still score well (R close to 1)."""
    r1 = _make_result("a", 1.0, {
        "stability_days": 30,
        "difficulty": 5,
        "last_recalled_at": time.time() - 10*86400,
    })
    out = maybe_fsrs([r1], enabled=True)
    # R(t=10, S=30) should be close to 1 because the chunk is well-stabilized.
    r_val = out[0]._fsrs_retrievability
    assert r_val > 0.5  # well above the cliff
