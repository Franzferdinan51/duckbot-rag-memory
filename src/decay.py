"""
decay.py — Ebbinghaus forgetting-curve weighting for the DuckBot brain.

Public-domain math: Hermann Ebbinghaus, "Memory: A Contribution to
Experimental Psychology" (1885). The retention function:

    R(t) = e^(-t / S)

where:
    t = elapsed time since the chunk was last recalled (or ingested)
    S = stability (a per-chunk float; increases with each successful recall)

This is Layer 8 of the brain-upgrade roadmap (docs/RESEARCH.md). Pattern
references (verified via GitHub REST API 2026-06-23):
  - YourMemory/sachitrafa/YourMemory — CC-BY-NC 4.0 (no code reuse, but
    validates the approach: +16pp LoCoMo recall vs mem0).
  - MemPalace/mempalace v4 — "Time-decay scoring" shipped in v4.0.0-alpha.

What this module does:
  - Computes R(t) per chunk given its age and stability.
  - Adjusts final_score during retrieval by multiplying with R(t).
  - Bumps stability on each successful recall (FSRS-6-lite: recall at
    scheduled time roughly doubles stability; miss penalizes it).
  - Provides a soft cap on retrieval: chunks with R(t) below `floor` are
    deprioritized, NOT deleted. Storage is append-only; we never lose
    history.

Design rules:
  - Default OFF. Opt-in via DUCKBOT_DECAY=1 env var or per-call `decay=True`.
  - No LLM call required. Pure math, microseconds per chunk.
  - Failure-safe: if anything throws, retrieval proceeds without decay.
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Iterable

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

# Stability floor (days). Chunks below this are still retrievable but heavily
# down-weighted. ~1 day = a fresh "I haven't seen this yet" memory.
DEFAULT_STABILITY_DAYS = 7.0

# Initial stability for a fresh chunk. MemPalace defaults to ~30 days for
# important items; we use 14 as a midpoint (between 1-day chatter and
# permanent rules).
INITIAL_STABILITY_DAYS = 14.0

# Below this retention score, retrieval treats the chunk as cold storage.
# Soft cap (chunk is still in the DB; just rarely surfaces).
DECAY_FLOOR = 0.05

# How much stability grows per successful recall. Ebbinghaus himself observed
# that recall at the right time roughly doubles retention. We use a milder
# factor (1.5x) so stability growth stays bounded; FSRS-6 uses similar.
RECALL_STABILITY_BOOST = 1.5

# How much stability shrinks on miss (i.e., the chunk was eligible to be
# recalled but wasn't). MemPalace v4 doesn't decay on miss (it only does
# time-based decay); we follow their lead and don't penalize misses.
# Set RECALL_MISS_PENALTY = 1.0 to opt into a penalty.
RECALL_MISS_PENALTY = 1.0


# -----------------------------------------------------------------------------
# Core math
# -----------------------------------------------------------------------------


def ebbinghaus_retention(
    age_days: float,
    stability_days: float = DEFAULT_STABILITY_DAYS,
) -> float:
    """Compute R(t) = e^(-t / S).

    Args:
        age_days: Days since the chunk was last recalled (or first ingested).
        stability_days: How "fixed" the memory is. Higher = decays slower.
            Doubles per successful recall (default).

    Returns:
        Retention in [0, 1]. 1.0 = perfectly fresh; 0.0 = completely forgotten.

    Public-domain math (Ebbinghaus 1885).
    """
    if stability_days <= 0:
        return 0.0
    if age_days < 0:
        age_days = 0.0
    return math.exp(-age_days / stability_days)


def days_since(epoch: float | int | None) -> float:
    """Convert a unix timestamp to days since then. Returns inf if epoch is None."""
    if epoch is None:
        return float("inf")
    now = time.time()
    return max(0.0, (now - float(epoch)) / 86400.0)


def bump_stability(
    current_stability: float | None,
    *,
    recalled: bool = True,
) -> float:
    """Update a chunk's stability based on a recall event.

    Args:
        current_stability: Current S value (or None for fresh chunks).
        recalled: True if the chunk was actually returned to the user;
            False if it was eligible but not picked (only used if
            RECALL_MISS_PENALTY != 1.0).

    Returns:
        New stability in days.
    """
    s = float(current_stability) if current_stability else INITIAL_STABILITY_DAYS
    if recalled:
        return s * RECALL_STABILITY_BOOST
    return s * RECALL_MISS_PENALTY


# -----------------------------------------------------------------------------
# Score blending
# -----------------------------------------------------------------------------


@dataclass
class DecayWeightedResult:
    """Result of decay-weighting a single chunk."""

    chunk_id: str
    age_days: float
    stability_days: float
    retention: float
    original_score: float
    decay_adjusted_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "age_days": round(self.age_days, 3),
            "stability_days": round(self.stability_days, 3),
            "retention": round(self.retention, 4),
            "original_score": round(self.original_score, 4),
            "decay_adjusted_score": round(self.decay_adjusted_score, 4),
        }


def decay_adjust(
    results: Iterable[Any],
    *,
    weight_decay: float = 0.4,
    weight_original: float = 0.6,
    decay_floor: float = DECAY_FLOOR,
    floor_penalty: float = 0.1,
) -> list[DecayWeightedResult]:
    """Apply Ebbinghaus retention weighting to a list of query results.

    Args:
        results: Iterable of objects with `.chunk_id`, `.rrf_score`,
            and `.metadata` containing `last_recalled_at` (or `ingested_at`),
            and optional `stability_days`.
        weight_decay: Multiplier for retention term.
        weight_original: Multiplier for original retrieval score.
            Final = weight_original * norm_original + weight_decay * retention.
        decay_floor: Retention below this gets an extra multiplicative
            penalty (default `floor_penalty`).
        floor_penalty: Multiplier applied to retention for cold chunks
            (R < decay_floor). Keeps them retrievable but very down-weighted.

    Returns:
        List of DecayWeightedResult sorted by decay_adjusted_score desc.
    """
    out: list[DecayWeightedResult] = []
    materialized = list(results)
    if not materialized:
        return out

    # Min-max normalize the original scores so the decay term doesn't get
    # crushed by tiny RRF values (which are typically 0.01..0.05).
    scores = []
    for r in materialized:
        try:
            scores.append(float(getattr(r, "rrf_score", 0.0) or 0.0))
        except Exception:
            scores.append(0.0)
    lo = min(scores) if scores else 0.0
    hi = max(scores) if scores else 0.0
    span = (hi - lo) if hi > lo else 1.0

    for r, s in zip(materialized, scores):
        meta = getattr(r, "metadata", {}) or {}
        # Prefer last_recalled_at (more accurate "last touched" timestamp).
        ts = meta.get("last_recalled_at") or meta.get("ingested_at")
        age = days_since(ts)
        if age == float("inf"):
            # No timestamp at all — treat as fully fresh (operator-imported).
            age = 0.0
        # Use an explicit `is None` check (not an `or` chain) so an
        # explicit stability_days=0 is respected as "unmemorized" rather
        # than silently upgraded to DEFAULT_STABILITY_DAYS via truthiness.
        _stab_raw = meta.get("stability_days")
        if _stab_raw is None:
            stability = DEFAULT_STABILITY_DAYS
        else:
            try:
                stability = float(_stab_raw)
            except (TypeError, ValueError):
                stability = DEFAULT_STABILITY_DAYS
        retention = ebbinghaus_retention(age, stability)
        if retention < decay_floor:
            retention *= floor_penalty

        norm = (s - lo) / span if span else 0.0
        adjusted = weight_original * norm + weight_decay * retention

        cid = getattr(r, "chunk_id", None) or ""
        out.append(
            DecayWeightedResult(
                chunk_id=cid,
                age_days=age,
                stability_days=stability,
                retention=retention,
                original_score=s,
                decay_adjusted_score=adjusted,
            )
        )

    out.sort(key=lambda d: d.decay_adjusted_score, reverse=True)
    return out


# -----------------------------------------------------------------------------
# Drop-in hook for src/query.py and src/connectors
# -----------------------------------------------------------------------------


def maybe_decay(
    results: list[Any],
    *,
    enabled: bool | None = None,
) -> list[Any]:
    """Apply decay weighting to a list of query results (in place if enabled).

    Args:
        results: List of objects with .chunk_id, .rrf_score, .metadata.
        enabled: True/False forces on/off. None reads DUCKBOT_DECAY env var.

    Returns:
        The input list, with .rrf_score replaced by decay_adjusted_score if
        enabled. Same list object if enabled=False.
    """
    if enabled is None:
        enabled = os.environ.get("DUCKBOT_DECAY", "0").lower() in ("1", "true", "yes")

    if not enabled or not results:
        return results

    weighted = decay_adjust(results)
    by_id = {w.chunk_id: w for w in weighted}
    for r in results:
        cid = getattr(r, "chunk_id", None)
        if cid and cid in by_id:
            w = by_id[cid]
            r.rrf_score = w.decay_adjusted_score
            # Stash raw retention for debugging.
            r.metadata = dict(r.metadata or {})
            r.metadata["decay_retention"] = round(w.retention, 4)
            r.metadata["decay_age_days"] = round(w.age_days, 3)
            r.metadata["decay_stability_days"] = round(w.stability_days, 3)
    results.sort(key=lambda r: r.rrf_score, reverse=True)
    return results


__all__ = [
    "DEFAULT_STABILITY_DAYS",
    "INITIAL_STABILITY_DAYS",
    "DECAY_FLOOR",
    "RECALL_STABILITY_BOOST",
    "ebbinghaus_retention",
    "days_since",
    "bump_stability",
    "DecayWeightedResult",
    "decay_adjust",
    "maybe_decay",
]
