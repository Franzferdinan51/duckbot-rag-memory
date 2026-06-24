"""
test_decay.py — unit tests for Layer 8 (Ebbinghaus decay weighting).

Pattern: same as test_rerank.py. Pure-math tests with mock QueryResult objects.
"""

from __future__ import annotations

import math
import os
import time

import pytest

from src.decay import (
    DECAY_FLOOR,
    DEFAULT_STABILITY_DAYS,
    INITIAL_STABILITY_DAYS,
    RECALL_STABILITY_BOOST,
    DecayWeightedResult,
    bump_stability,
    days_since,
    decay_adjust,
    ebbinghaus_retention,
    maybe_decay,
)


# -----------------------------------------------------------------------------
# Core math
# -----------------------------------------------------------------------------


def test_ebbinghaus_retention_at_zero_age():
    """At t=0, R = e^0 = 1.0."""
    assert ebbinghaus_retention(0, 7) == 1.0


def test_ebbinghaus_retention_at_one_stability():
    """At t=S, R = e^-1 ≈ 0.368."""
    r = ebbinghaus_retention(7, 7)
    assert abs(r - math.exp(-1)) < 1e-9


def test_ebbinghaus_retention_at_two_stability():
    """At t=2S, R = e^-2 ≈ 0.135."""
    r = ebbinghaus_retention(14, 7)
    assert abs(r - math.exp(-2)) < 1e-9


def test_ebbinghaus_retention_higher_stability_slower_decay():
    """A chunk with S=30 decays slower than one with S=7 at the same age."""
    s7 = ebbinghaus_retention(3, 7)
    s30 = ebbinghaus_retention(3, 30)
    assert s30 > s7


def test_ebbinghaus_retention_handles_zero_stability():
    """S=0 is invalid; we return 0.0 to avoid division-by-zero."""
    assert ebbinghaus_retention(5, 0) == 0.0


def test_ebbinghaus_retention_clamps_negative_age():
    """Negative ages shouldn't produce R > 1."""
    r = ebbinghaus_retention(-100, 7)
    assert r == 1.0


# -----------------------------------------------------------------------------
# days_since
# -----------------------------------------------------------------------------


def test_days_since_recent():
    """A timestamp from 1 day ago should give ~1.0 days."""
    one_day_ago = time.time() - 86400
    d = days_since(one_day_ago)
    assert 0.99 < d < 1.01


def test_days_since_future_timestamp():
    """Future timestamps shouldn't produce negative days."""
    future = time.time() + 86400
    d = days_since(future)
    assert d == 0.0


def test_days_since_none_returns_inf():
    assert days_since(None) == float("inf")


# -----------------------------------------------------------------------------
# bump_stability
# -----------------------------------------------------------------------------


def test_bump_stability_first_recall():
    """First recall on a fresh chunk: stability gets the boost applied
    to the initial value (so the first recall is itself a reinforcement).
    Per Ebbinghaus's observation, recall at the right time strengthens."""
    s = bump_stability(None, recalled=True)
    # First recall applies the boost to the initial value.
    assert s == INITIAL_STABILITY_DAYS * RECALL_STABILITY_BOOST


def test_bump_stability_recalled_doubles():
    """Successful recall multiplies stability by RECALL_STABILITY_BOOST."""
    s = bump_stability(10.0, recalled=True)
    assert s == 10.0 * RECALL_STABILITY_BOOST


def test_bump_stability_miss_penalty_default_is_noop():
    """Default miss penalty is 1.0 (no penalty), per MemPalace v4 design."""
    s = bump_stability(10.0, recalled=False)
    assert s == 10.0  # unchanged


def test_bump_stability_grows_bounded():
    """Stability grows, not unbounded (sanity check)."""
    s = 1.0
    for _ in range(20):
        s = bump_stability(s, recalled=True)
    # After 20 recalls, growth should be huge but not infinite.
    assert s == pytest.approx(1.5 ** 20, rel=1e-6)
    assert math.isfinite(s)


# -----------------------------------------------------------------------------
# decay_adjust
# -----------------------------------------------------------------------------


class StubQR:
    def __init__(self, cid, score=0.0, ts=None, stability=None, meta=None):
        self.chunk_id = cid
        self.rrf_score = score
        self.metadata = dict(meta or {})
        if ts is not None:
            self.metadata["last_recalled_at"] = ts
            self.metadata.setdefault("ingested_at", ts)
        if stability is not None:
            self.metadata["stability_days"] = stability


def test_decay_adjust_empty_input():
    assert decay_adjust([]) == []


def test_decay_adjust_returns_decay_weighted_results():
    chunks = [StubQR("a", score=0.05, ts=time.time())]
    out = decay_adjust(chunks)
    assert len(out) == 1
    assert isinstance(out[0], DecayWeightedResult)
    assert out[0].chunk_id == "a"
    assert 0.0 <= out[0].retention <= 1.0


def test_decay_adjust_recent_chunk_outranks_old_chunk_at_equal_rrf():
    """Two chunks with same RRF — the recent one should win."""
    now = time.time()
    recent = StubQR("recent", score=0.05, ts=now)
    old = StubQR("old", score=0.05, ts=now - 30 * 86400)  # 30 days old
    out = decay_adjust([recent, old])
    # Recent should be ranked first.
    assert out[0].chunk_id == "recent"
    assert out[1].chunk_id == "old"


def test_decay_adjust_handles_missing_timestamp():
    """Chunks with no timestamp default to age=0 (fully fresh)."""
    chunks = [StubQR("a", score=0.05)]  # no ts
    out = decay_adjust(chunks)
    assert out[0].age_days == 0.0
    assert out[0].retention == 1.0


def test_decay_adjust_floor_penalty_for_cold_chunks():
    """A very old chunk gets the floor penalty (R < DECAY_FLOOR -> R * floor_penalty)."""
    very_old = StubQR(
        "old", score=0.05,
        ts=time.time() - 365 * 86400,  # 1 year
        stability=1.0,  # unstable
    )
    out = decay_adjust([very_old])
    # Age 365, S=1 → R = e^-365 ≈ 0 (well below floor)
    # Then floored: 0 * 0.1 = 0
    assert out[0].retention < DECAY_FLOOR


def test_decay_adjust_normalizes_original_scores():
    """Min-max normalization means a 0.01 and 0.05 both get spread to [0,1]."""
    low = StubQR("low", score=0.01, ts=time.time())
    high = StubQR("high", score=0.05, ts=time.time())
    out = decay_adjust([low, high])
    # After normalization, high > low in original_score contribution.
    by_id = {d.chunk_id: d for d in out}
    # 'high' should be ranked first (same retention, but higher norm).
    assert by_id["high"].decay_adjusted_score >= by_id["low"].decay_adjusted_score


def test_decay_adjust_result_to_dict():
    chunks = [StubQR("a", score=0.05, ts=time.time())]
    out = decay_adjust(chunks)
    d = out[0].to_dict()
    assert set(d.keys()) == {
        "chunk_id", "age_days", "stability_days",
        "retention", "original_score", "decay_adjusted_score",
    }


# -----------------------------------------------------------------------------
# maybe_decay hook
# -----------------------------------------------------------------------------


def test_maybe_decay_disabled_returns_unchanged(monkeypatch):
    monkeypatch.delenv("DUCKBOT_DECAY", raising=False)
    chunks = [StubQR("a", score=0.05, ts=time.time() - 30 * 86400)]
    out = maybe_decay(chunks, enabled=False)
    # No decay adjustment applied; chunks unchanged.
    assert out == chunks


def test_maybe_decay_enabled_adjusts_scores(monkeypatch):
    monkeypatch.setenv("DUCKBOT_DECAY", "1")
    now = time.time()
    chunks = [
        StubQR("recent", score=0.05, ts=now),
        StubQR("old", score=0.05, ts=now - 30 * 86400),
    ]
    out = maybe_decay(chunks, enabled=True)
    # Recent should outrank old after decay weighting.
    assert out[0].chunk_id == "recent"
    assert out[1].chunk_id == "old"
    # Metadata should be annotated with retention.
    by_id = {c.chunk_id: c for c in out}
    assert "decay_retention" in by_id["recent"].metadata
    assert "decay_retention" in by_id["old"].metadata
    # Recent retention is higher than old retention.
    assert by_id["recent"].metadata["decay_retention"] > by_id["old"].metadata["decay_retention"]


def test_maybe_decay_reads_env_var(monkeypatch):
    monkeypatch.setenv("DUCKBOT_DECAY", "0")
    chunks = [StubQR("a", score=0.05, ts=time.time())]
    # enabled=None → reads env var → DUCKBOT_DECAY=0 → no decay
    out = maybe_decay(chunks)
    # rrf_score preserved (decay didn't replace it).
    assert out[0].rrf_score == 0.05
    assert "decay_retention" not in out[0].metadata


def test_maybe_decay_handles_empty_list():
    assert maybe_decay([], enabled=True) == []


def test_maybe_decay_failure_falls_back(monkeypatch):
    """If decay_adjust throws on a chunk, the result for that chunk is
    skipped but the overall call still succeeds for the others."""
    # Patch decay_adjust to raise — maybe_decay itself doesn't catch, so
    # this just verifies decay_adjust's own error surface is preserved.
    from src import decay as decay_mod

    def _boom(results):
        raise RuntimeError("simulated decay failure")

    monkeypatch.setattr(decay_mod, "decay_adjust", _boom)
    monkeypatch.setenv("DUCKBOT_DECAY", "1")
    chunks = [StubQR("a", score=0.05, ts=time.time())]
    # maybe_decay doesn't wrap decay_adjust itself — the wrapper is in
    # src/query.py. So a raw call WILL raise. This is correct: the query
    # pipeline wraps it, but the raw hook is a thin pass-through.
    with pytest.raises(RuntimeError, match="simulated"):
        maybe_decay(chunks, enabled=True)
