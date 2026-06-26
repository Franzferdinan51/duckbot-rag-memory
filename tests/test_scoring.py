"""Tests for src/scoring.py — 5-factor priority scoring.

Pure functions; no I/O. We pin `now` for determinism so the recency
factor is reproducible.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Repo root on sys.path so `src.scoring` is importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.scoring import (  # noqa: E402
    W_RECENCY,
    W_FREQUENCY,
    W_CONNECTIVITY,
    W_EXPLICIT,
    W_TYPE,
    TIER_WEIGHT,
    recency_factor,
    frequency_factor,
    connectivity_factor,
    explicit_factor,
    type_factor,
    priority_score,
    sort_by_priority,
)


# Fixed reference time so all time-based math is deterministic.
NOW = 1_780_000_000.0  # 2026-06-15-ish


# ---------------------------------------------------------------------------
# Factor weights — sanity check (they MUST sum to 1.0; the whole formula
# assumes this and the docs quote the same numbers).
# ---------------------------------------------------------------------------


def test_weights_sum_to_one():
    assert abs((W_RECENCY + W_FREQUENCY + W_CONNECTIVITY + W_EXPLICIT + W_TYPE) - 1.0) < 1e-9


def test_weights_match_documented_values():
    assert W_RECENCY == 0.30
    assert W_FREQUENCY == 0.25
    assert W_CONNECTIVITY == 0.20
    assert W_EXPLICIT == 0.15
    assert W_TYPE == 0.10


# ---------------------------------------------------------------------------
# Recency factor
# ---------------------------------------------------------------------------


def test_recency_factor_zero_age_returns_one():
    """Chunk accessed right now → recency = 1.0."""
    assert recency_factor(NOW, NOW, now=NOW) == pytest.approx(1.0, abs=1e-9)


def test_recency_factor_half_life_returns_half():
    """At half-life (30 days) → recency = 0.5."""
    half_life_ago = NOW - 30 * 86400
    assert recency_factor(half_life_ago, None, now=NOW) == pytest.approx(0.5, abs=1e-3)


def test_recency_factor_falls_back_to_ingested_at():
    """When last_recalled_at is missing, use ingested_at."""
    ingested = NOW - 5 * 86400  # 5 days ago
    assert recency_factor(None, ingested, now=NOW) == recency_factor(ingested, None, now=NOW)


def test_recency_factor_missing_timestamps_returns_midpoint():
    """No timestamps → 0.5 (don't drown missing-data chunks)."""
    assert recency_factor(None, None, now=NOW) == 0.5
    assert recency_factor(0, None, now=NOW) == 0.5


def test_recency_factor_monotonically_decreasing():
    """Older chunks should never score higher than newer ones."""
    scores = [
        recency_factor(NOW - d * 86400, None, now=NOW)
        for d in [0, 1, 7, 30, 90, 365]
    ]
    assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Frequency factor
# ---------------------------------------------------------------------------


def test_frequency_factor_zero_recalls_returns_zero():
    assert frequency_factor(0) == 0.0


def test_frequency_factor_saturates_at_100():
    assert frequency_factor(100) == pytest.approx(1.0, abs=1e-9)
    assert frequency_factor(1000) == pytest.approx(1.0, abs=1e-9)


def test_frequency_factor_log_scaled():
    """10 recalls should give ~ log10(11)/log10(101) ≈ 0.52."""
    assert frequency_factor(10) == pytest.approx(math.log10(11) / math.log10(101), abs=1e-9)


# ---------------------------------------------------------------------------
# Connectivity factor
# ---------------------------------------------------------------------------


def test_connectivity_factor_zero_zero_returns_zero():
    assert connectivity_factor(0, 0) == 0.0


def test_connectivity_factor_edges_dominate_entities():
    """Edges count 60% in the weighted average; entities 40%."""
    e_only = connectivity_factor(entity_count=10, edge_count=0)
    g_only = connectivity_factor(entity_count=0, edge_count=20)
    # Both should be near 1.0 but e_only and g_only contribute differently.
    assert e_only == pytest.approx(0.4 * 1.0 + 0.6 * 0.0, abs=1e-9)
    assert g_only == pytest.approx(0.4 * 0.0 + 0.6 * 1.0, abs=1e-9)


def test_connectivity_factor_caps_at_one():
    """Large entity/edge counts don't exceed 1.0."""
    assert connectivity_factor(entity_count=1000, edge_count=1000) == 1.0


# ---------------------------------------------------------------------------
# Explicit + type factors
# ---------------------------------------------------------------------------


def test_explicit_factor_clamps_to_unit_interval():
    assert explicit_factor(None) == 0.5  # neutral default
    assert explicit_factor(0.0) == 0.0
    assert explicit_factor(1.0) == 1.0
    assert explicit_factor(2.5) == 1.0  # clamped
    assert explicit_factor(-0.5) == 0.0  # clamped


def test_type_factor_tier_ranking():
    """Procedural > semantic > episodic > working."""
    assert type_factor("procedural") > type_factor("semantic")
    assert type_factor("semantic") > type_factor("episodic")
    assert type_factor("episodic") > type_factor("working")


def test_type_factor_unknown_tier_returns_neutral():
    assert type_factor(None) == 0.5
    assert type_factor("bogus") == 0.5


def test_type_factor_table_matches_weights():
    """TIER_WEIGHT must agree with the formula above."""
    for tier, expected in [
        ("procedural", 1.00),
        ("semantic",   0.90),
        ("episodic",   0.70),
        ("working",    0.50),
    ]:
        assert TIER_WEIGHT[tier] == expected


# ---------------------------------------------------------------------------
# Composite priority_score
# ---------------------------------------------------------------------------


def _chunk_meta(**kwargs) -> dict:
    """Build a chunk-meta dict with sensible defaults."""
    base = {
        "tier": "semantic",
        "importance": 0.5,
        "recall_count": 0,
        "ingested_at": NOW - 86400,
        "last_recalled_at": None,
    }
    base.update(kwargs)
    return base


def test_priority_score_in_unit_interval():
    """All factor scores are 0..1 so the composite must also be 0..1."""
    for meta in [
        _chunk_meta(),
        _chunk_meta(tier="procedural", importance=1.0, recall_count=1000,
                    last_recalled_at=NOW),
        _chunk_meta(tier="working", importance=0.0, recall_count=0),
    ]:
        score = priority_score(meta, now=NOW)
        assert 0.0 <= score <= 1.0


def test_priority_score_fresh_recall_ranks_higher_than_old():
    """A chunk just recalled beats one recalled a year ago."""
    fresh = _chunk_meta(last_recalled_at=NOW, ingested_at=NOW - 86400)
    old = _chunk_meta(last_recalled_at=NOW - 365 * 86400, ingested_at=NOW - 365 * 86400)
    assert priority_score(fresh, now=NOW) > priority_score(old, now=NOW)


def test_priority_score_high_frequency_ranks_higher():
    """A frequently-recalled chunk beats one never recalled (same recency)."""
    base = _chunk_meta(last_recalled_at=NOW - 86400, ingested_at=NOW - 86400)
    hot = dict(base, recall_count=50)
    cold = dict(base, recall_count=0)
    assert priority_score(hot, now=NOW) > priority_score(cold, now=NOW)


def test_priority_score_high_importance_ranks_higher():
    """User-set importance directly affects ranking."""
    base = _chunk_meta()
    high = dict(base, importance=1.0)
    low = dict(base, importance=0.1)
    assert priority_score(high, now=NOW) > priority_score(low, now=NOW)


def test_priority_score_procedural_beats_working():
    """Type factor: procedural chunks win over working-tier ones."""
    base = _chunk_meta()
    proc = dict(base, tier="procedural")
    work = dict(base, tier="working")
    assert priority_score(proc, now=NOW) > priority_score(work, now=NOW)


def test_priority_score_connectivity_via_graph_params():
    """entity_count + edge_count feed the connectivity factor."""
    base = _chunk_meta()
    iso = priority_score(base, entity_count=0, edge_count=0, now=NOW)
    hub = priority_score(base, entity_count=5, edge_count=10, now=NOW)
    assert hub > iso


def test_priority_score_handles_missing_metadata_gracefully():
    """Empty / partial metadata should not crash — returns something in [0,1]."""
    for meta in [{}, {"tier": None}, {"importance": "garbage"}, {"recall_count": "many"}]:
        score = priority_score(meta, now=NOW)
        assert 0.0 <= score <= 1.0


def test_priority_score_weight_consistency():
    """A chunk with all-factors-maxed should approach W_TIER * 1.0
    + W_EXPLICIT * 1.0 + W_RECENCY * 1.0 = 0.55 (recency=1, explicit=1,
    type=procedural=1, frequency=1, connectivity=1)."""
    meta = _chunk_meta(
        tier="procedural",
        importance=1.0,
        recall_count=1000,
        last_recalled_at=NOW,
    )
    score = priority_score(
        meta,
        entity_count=10,
        edge_count=20,
        now=NOW,
    )
    # All five factors at 1.0 → composite = 1.0.
    assert score == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# sort_by_priority (integration with chunk dict shape)
# ---------------------------------------------------------------------------


def _chunk_dict(chunk_id: str, **meta_overrides) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": f"text for {chunk_id}",
        "tier": "semantic",
        "importance": 0.5,
        "score": 0.5,
        "source_path": f"/tmp/{chunk_id}.md",
        "metadata": _chunk_meta(**meta_overrides),
    }


def test_sort_by_priority_orders_highest_score_first():
    chunks = [
        _chunk_dict("cold", recall_count=0, ingested_at=NOW - 365 * 86400),
        _chunk_dict("hot",  recall_count=50, last_recalled_at=NOW - 86400,
                    ingested_at=NOW - 86400),
        _chunk_dict("fresh", importance=1.0, last_recalled_at=NOW,
                    ingested_at=NOW - 86400),
    ]
    out = sort_by_priority(chunks, now=NOW)
    # "hot" wins: recency≈1.0 + frequency (log10(51)/log10(101)≈0.85)
    # gives 0.30+0.21+0+0.075+0.09=0.68 vs "fresh"'s 0.30+0+0+0.15+0.09=0.54.
    # "cold" (recency exp(-ln2*365/30)≈0) is at the bottom regardless.
    assert out[0]["chunk_id"] == "hot"
    assert out[-1]["chunk_id"] == "cold"
    assert out[1]["chunk_id"] == "fresh"


def test_sort_by_priority_does_not_mutate_input():
    chunks = [
        _chunk_dict("a"),
        _chunk_dict("b"),
    ]
    snapshot = list(chunks)
    sort_by_priority(chunks, now=NOW)
    assert chunks == snapshot


def test_sort_by_priority_handles_missing_chunk_id():
    """A chunk with no chunk_id gets the lowest score (sink)."""
    chunks = [
        _chunk_dict("real"),
        {"text": "no id", "metadata": _chunk_meta()},  # no chunk_id
    ]
    out = sort_by_priority(chunks, now=NOW)
    assert out[0]["chunk_id"] == "real"
    assert out[1].get("chunk_id") is None


def test_sort_by_priority_uses_metadata_key_by_default():
    """Override `key_meta` lets non-standard dicts sort correctly."""
    chunks = [
        {"chunk_id": "a", "raw_meta": _chunk_meta(importance=1.0)},
        {"chunk_id": "b", "raw_meta": _chunk_meta(importance=0.1)},
    ]
    out = sort_by_priority(chunks, key_meta="raw_meta", now=NOW)
    assert out[0]["chunk_id"] == "a"
    assert out[1]["chunk_id"] == "b"