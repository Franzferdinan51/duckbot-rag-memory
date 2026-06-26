"""scoring.py — 5-factor importance scoring for brain_wake_up ranking.

Inspired by MindBank's importance formula (spfcraze/MindBank) but
re-implemented natively against duckbot's data model. The five factors:

    Factor        Weight  Source
    ────────────  ──────  ─────────────────────────────────────────────
    Recency       30%     time since last_recalled_at (or ingested_at)
    Frequency     25%     log-scaled recall_count
    Connectivity  20%     graph degree (entity_count + edge_count)
    Explicit      15%     user-set importance (0..1, stored in metadata)
    Type          10%     tier weight (procedural > semantic > episodic > working)

The composite score is a 0..1 float used to re-rank wake_up results
before they're returned to the agent. Higher = more important.

Pure functions; no I/O. The Brain facade wires them into wake_up
(via `_priority_score` in src/connectors/base.py) so the scoring is
available wherever chunk metadata is.

Licensed MIT (this file is original work; the formula is borrowed
from MindBank's design, which is also MIT).
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional

# Factor weights (sum = 1.0). Keep these in sync with the table in
# docs/PLUGIN_SURFACE.md so docs and code agree.
W_RECENCY = 0.30
W_FREQUENCY = 0.25
W_CONNECTIVITY = 0.20
W_EXPLICIT = 0.15
W_TYPE = 0.10

# Tier weight table — 10% of the composite.
TIER_WEIGHT: dict[str, float] = {
    "procedural": 1.00,  # rules + how-tos; load these first
    "semantic":   0.90,  # facts + knowledge
    "episodic":   0.70,  # past events / sessions
    "working":    0.50,  # current-session context (transient)
}

# Recency decay: how many days back does the recency factor cover?
# Chunks older than this cap at 0 (but still get the other factors).
RECENCY_HALF_LIFE_DAYS = 30.0


def _safe_float(v: Any, default: float = 0.0) -> float:
    """Coerce metadata values to float without crashing on garbage."""
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v: Any, default: int = 0) -> int:
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def recency_factor(
    last_recalled_at: Optional[float],
    ingested_at: Optional[float],
    now: Optional[float] = None,
) -> float:
    """0..1 — higher when the chunk was accessed recently.

    Uses exponential decay with a 30-day half-life. Chunks never
    recalled fall back to ingested_at. Chunks with no timestamps
    return the midpoint (0.5) so they don't drown.
    """
    import time as _time
    now = now if now is not None else _time.time()
    ts = last_recalled_at or ingested_at or 0.0
    if ts <= 0:
        return 0.5
    age_days = max(0.0, (now - ts) / 86400.0)
    # exp(-ln(2) * age / half_life) — same shape as MindBank's recency
    return math.exp(-math.log(2) * age_days / RECENCY_HALF_LIFE_DAYS)


def frequency_factor(recall_count: int) -> float:
    """0..1 — log-scaled frequency of recall.

    log10(1 + count) / log10(1 + 100) saturates at ~1.0 once a chunk
    has been recalled ~100 times (which would mean it's a hotspot).
    """
    if recall_count <= 0:
        return 0.0
    return min(1.0, math.log10(1 + recall_count) / math.log10(101))


def connectivity_factor(
    entity_count: int = 0,
    edge_count: int = 0,
) -> float:
    """0..1 — how well-connected this chunk is in the graph.

    Combines entities extracted at ingest time (stored in metadata)
    with the chunk's edge count from the entity graph. Both are
    capped so a single chunk with hundreds of edges doesn't
    dominate — we want "well-connected", not "maximal".
    """
    # 5 entities or 10 edges → saturates at ~1.0
    e_score = min(1.0, entity_count / 5.0)
    g_score = min(1.0, edge_count / 10.0)
    # Weighted average — edges count a bit more (they're deliberate).
    return 0.4 * e_score + 0.6 * g_score


def explicit_factor(importance: float) -> float:
    """0..1 — user-set importance, clamped to [0, 1]."""
    if importance is None:
        return 0.5  # neutral default
    return max(0.0, min(1.0, float(importance)))


def type_factor(tier: Optional[str]) -> float:
    """0..1 — tier weight (procedural > semantic > episodic > working)."""
    if not tier:
        return 0.5
    return TIER_WEIGHT.get(str(tier).lower(), 0.5)


def priority_score(
    chunk_meta: Mapping[str, Any],
    *,
    entity_count: int = 0,
    edge_count: int = 0,
    now: Optional[float] = None,
) -> float:
    """Composite priority score 0..1 for a chunk.

    Args:
      chunk_meta: the chunk's metadata dict (must include at least
        `tier` for meaningful output; the other factors degrade
        gracefully if fields are missing).
      entity_count: number of entities extracted from this chunk
        (from the graph entity table). Default 0.
      edge_count: number of graph edges where this chunk is an endpoint.
        Default 0.
      now: timestamp for "now" (defaults to time.time(); pass an
        explicit value in tests for determinism).

    Returns:
      float in [0.0, 1.0]. Higher = more important.
    """
    md = chunk_meta or {}
    r = recency_factor(
        _safe_float(md.get("last_recalled_at")) or None,
        _safe_float(md.get("ingested_at")) or None,
        now=now,
    )
    f = frequency_factor(_safe_int(md.get("recall_count")))
    c = connectivity_factor(
        entity_count=entity_count,
        edge_count=edge_count,
    )
    e = explicit_factor(_safe_float(md.get("importance"), default=0.5))
    t = type_factor(md.get("tier"))

    score = (
        W_RECENCY * r
        + W_FREQUENCY * f
        + W_CONNECTIVITY * c
        + W_EXPLICIT * e
        + W_TYPE * t
    )
    # Clamp to [0, 1] for safety (the factors are already in range but
    # defense in depth — a bad upstream value shouldn't blow up the
    # sort key).
    return max(0.0, min(1.0, score))


def sort_by_priority(
    chunks: list[dict],
    *,
    key_meta: str = "metadata",
    key_chunk_id: str = "chunk_id",
    now: Optional[float] = None,
) -> list[dict]:
    """Sort a list of chunk dicts by priority_score descending.

    Convenience wrapper — each dict must have the metadata under
    `metadata` (or override via `key_meta`). Returns a NEW list;
    doesn't mutate the input.

    Designed for use inside `Brain.wake_up` after the recall pass:
    results come back ranked by RRF (semantic similarity); we
    re-rank by priority so an old-but-frequently-recalled
    high-importance chunk surfaces above a fresh-but-low-importance
    noise chunk.
    """
    def _key(c: dict) -> float:
        meta = c.get(key_meta) or {}
        if not isinstance(meta, Mapping):
            meta = {}
        # Chunks that are missing chunk_id get the lowest priority
        # (shouldn't happen in practice — wake_up always assigns).
        if not c.get(key_chunk_id):
            return -1.0
        return priority_score(meta, now=now)

    return sorted(chunks, key=_key, reverse=True)


__all__ = [
    "W_RECENCY",
    "W_FREQUENCY",
    "W_CONNECTIVITY",
    "W_EXPLICIT",
    "W_TYPE",
    "TIER_WEIGHT",
    "recency_factor",
    "frequency_factor",
    "connectivity_factor",
    "explicit_factor",
    "type_factor",
    "priority_score",
    "sort_by_priority",
]