"""
fsrs.py — Layer 9: FSRS-6 spaced repetition math.

Reimplements the Free Spaced Repetition Scheduler v6 algorithm
(https://github.com/open-spaced-repetition/fsrs4anki/wiki/Free-SRS-Algorithm)
from spec, NOT from any source code. The algorithm is described
publicly in the FSRS-6 paper and AnKing's open documentation.

Why not vendor? The reference implementations (ts-fsrs, py-fsrs,
rs-fsrs) are mostly MIT, but the canonical AnKing-maintained
py-fsrs carries a complex changelog and is tightly coupled to
Anki's review state machine — too much surface to bring in.
The algorithm itself is just math; we reimplement the parts we
need (stability growth + retrievability) and skip the scheduler.

Three key concepts (FSRS-6 spec):
  - Stability (S): the interval at which retention is ~90%.
  - Difficulty (D): 1.0 (easiest) to 10.0 (hardest). 5.0 default.
  - Retrievability (R): probability of recall right now (0-1).

Public-domain math from the FSRS-6 spec:
  R(t, S)      = (1 + factor * t / S) ^ (1 / factor)
                where factor = 19/81 (FSRS-6 default)
  S'_r         = S * (e^w8 * (11 - D) * S^(-0.8) * (1 - R) + 1)
                where w8 is a per-deployment weight (default 0.02)
  D'_r         = D - w6 * (R - 0.5)
                where w6 is a per-deployment weight (default 0.1)

Compared to L8 (Ebbinghaus):
  - Ebbinghaus R = e^(-t/S)  (assumes S is a single half-life)
  - FSRS-6 R      = power law with factor 19/81 (matches real forgetting curves better)
  - Ebbinghaus stability growth: fixed 1.5× on recall
  - FSRS-6 stability growth: depends on D, S, R — easy + well-learned chunks
    grow slower; hard + rarely-recalled chunks grow faster
  - Ebbinghaus: no notion of difficulty
  - FSRS-6: explicit difficulty tracking per chunk

Opt-in. Default OFF (L8's simpler math is the default).
Enable via:
  - DUCKBOT_FSRS=1
  - fsrs=True kwarg to Brain.recall() / Memory.recall()
  - per-chunk: pass chunk.metadata["fsrs"] = True
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from typing import Any


# -----------------------------------------------------------------------------
# FSRS-6 weights (from the spec; per-deployment overrides possible)
# -----------------------------------------------------------------------------

# Default decay exponent. The actual FSRS-6 formula (per AnKing's open
# documentation) is:
#   R(t, S) = (1 + t / (9 * S)) ^ (-w20)
# where w20 is the per-deployment decay exponent. The published FSRS-6
# default is w20 = 0.1542 for the reference 17-weight set; we use a
# more aggressive 0.9 by default because our chunks are denser (more
# knowledge per item) and a 7-day stability should give ~90% recall,
# not 98% — we want to actually down-weight old chunks.
#
# This is the only FSRS-6 weight we expose; deployments tune it.
DEFAULT_W20 = 0.9

# Default stability-growth weight (FSRS-6 spec): w8 = 0.02 in the
# reference implementation. Higher = stability grows faster on recall.
DEFAULT_W8 = 0.02

# Default difficulty-update weight (FSRS-6 spec): w6 = 0.1.
# Higher = difficulty changes more aggressively with each recall.
DEFAULT_W6 = 0.1

# Default difficulty bounds: 1.0 (easiest) to 10.0 (hardest).
DIFFICULTY_MIN = 1.0
DIFFICULTY_MAX = 10.0

# Default difficulty for fresh chunks (FSRS-6 spec).
DEFAULT_DIFFICULTY = 5.0

# Stability bounds (FSRS-6 spec). Below STABILITY_FLOOR we'd consider
# the chunk effectively un-memorized.
STABILITY_FLOOR = 0.01
STABILITY_CEILING = 36500.0  # 100 years; effectively unlimited

# Retrievability bounds.
R_MIN = 0.0
R_MAX = 1.0


# -----------------------------------------------------------------------------
# Core math
# -----------------------------------------------------------------------------


def fsrs_retrievability(
    elapsed_days: float,
    stability: float,
    w20: float = DEFAULT_W20,
) -> float:
    """FSRS-6 power-law retrievability (AnKing form).

    R(t, S) = (1 + t / (9 * S)) ^ (-w20)
             clamped to [0, 1].

    At t = 0: R = 1.0 (just recalled).
    At t = 9S: R = (1 + 1)^(-w20) = 2^(-w20) ≈ 0.535 with w20=0.9.
    At t → ∞: R → 0.

    The exponent w20 controls the steepness of the curve. Larger w20 =
    steeper forgetting. The published FSRS-6 default is w20 = 0.1542;
    we ship 0.9 as a sane default for our dense knowledge chunks.

    Args:
        elapsed_days: days since last review.
        stability: chunk's stability in days (S).
        w20: FSRS-6 decay exponent (default 0.9).

    Returns:
        Probability of recall right now, in [0, 1].
    """
    if stability <= 0:
        return 0.0
    if elapsed_days < 0:
        elapsed_days = 0.0
    if w20 <= 0:
        # w20 is the decay exponent. w20=0 means R = base**0 = 1 (no
        # forgetting), and is the "disable decay" setting — NOT "no
        # recall". Negative is meaningless but should also mean "never
        # forget" rather than misreport the math.
        return R_MAX
    try:
        base = 1.0 + elapsed_days / (9.0 * stability)
        if base <= 0:
            return 0.0
        r = base ** (-w20)
    except (OverflowError, ValueError):
        return 0.0
    return max(R_MIN, min(R_MAX, r))


def fsrs_bump_stability(
    *,
    stability: float | None,
    difficulty: float,
    retrievability: float,
    recalled: bool = True,
    w8: float = DEFAULT_W8,
) -> float:
    """FSRS-6 stability update after a recall attempt.

    S'_r = S * (e^w8 * (11 - D) * S^(-0.8) * (1 - R) + 1)

    Args:
        stability: current S in days (None for fresh chunks; uses default 7).
        difficulty: current D in [1, 10].
        retrievability: current R in [0, 1] (the R just before this recall).
        recalled: True if user got it right; False if recall failed.
        w8: FSRS-6 weight (default 0.02).

    Returns:
        New stability in days, clamped to [STABILITY_FLOOR, STABILITY_CEILING].
    """
    s = stability if (stability and stability > 0) else 7.0
    d = max(DIFFICULTY_MIN, min(DIFFICULTY_MAX, difficulty))
    r = max(R_MIN, min(R_MAX, retrievability))

    if recalled:
        # FSRS-6 success formula: S grows by a multiplier that depends on
        # D, S, and (1 - R). Easy chunks (low D) and recently-recalled
        # chunks (high R → small (1-R)) grow slower; hard + forgotten
        # chunks grow faster.
        try:
            multiplier = (
                math.exp(w8)
                * (11.0 - d)
                * (s ** (-0.8))
                * (1.0 - r)
                + 1.0
            )
        except (OverflowError, ValueError):
            multiplier = 1.0
        new_s = s * multiplier
    else:
        # FSRS-6 failure formula: S shrinks. From the spec, the canonical
        # failure update is S' = S / 2 * (1 + w8 * (R - 0.5)).
        # We reimplement the *spirit* (S shrinks more if you were
        # confident; less if you weren't), clamped at STABILITY_FLOOR.
        try:
            new_s = s * 0.5 * (1.0 + w8 * (r - 0.5))
        except (OverflowError, ValueError):
            new_s = s * 0.5

    return max(STABILITY_FLOOR, min(STABILITY_CEILING, new_s))


def fsrs_bump_difficulty(
    *,
    difficulty: float,
    retrievability: float,
    recalled: bool = True,
    w6: float = DEFAULT_W6,
) -> float:
    """FSRS-6 difficulty update after a recall attempt.

    D'_r = D - w6 * (R - 0.5)  when recalled (positive sign makes higher R
                                                       decrease difficulty)
    D'_r = D + w6 * (1 - R)     when failed   (penalty: low R → large increase)

    Successful recall with R < 0.5 (you barely passed) → difficulty increases
    (you found it hard). Successful recall with R > 0.5 (you were confident)
    → difficulty decreases (you found it easy). Failed recall always
    increases difficulty, with the size of the increase inversely
    proportional to R (failed-with-low-R is much harder than failed-with-high-R).

    Args:
        difficulty: current D in [1, 10].
        retrievability: current R in [0, 1].
        recalled: True if user got it right.
        w6: FSRS-6 weight (default 0.1).

    Returns:
        New difficulty in [DIFFICULTY_MIN, DIFFICULTY_MAX].
    """
    d = max(DIFFICULTY_MIN, min(DIFFICULTY_MAX, difficulty))
    r = max(R_MIN, min(R_MAX, retrievability))

    if recalled:
        # Confident success (R > 0.5) → D decreases.
        # Barely-passed success (R < 0.5) → D increases.
        new_d = d - w6 * (r - 0.5)
    else:
        # Failed recall: D increases, more if R was low (you "almost had it").
        new_d = d + w6 * (1.0 - r)

    return max(DIFFICULTY_MIN, min(DIFFICULTY_MAX, new_d))


# -----------------------------------------------------------------------------
# Score adjustment (FSRS-6 + tier-prior compatible)
# -----------------------------------------------------------------------------


@dataclass
class FSRSWeightedResult:
    """Result of FSRS-6 weighting a single chunk."""

    chunk_id: str
    elapsed_days: float
    stability_days: float
    difficulty: float
    retrievability: float
    original_score: float
    fsrs_adjusted_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "elapsed_days": self.elapsed_days,
            "stability_days": self.stability_days,
            "difficulty": self.difficulty,
            "retrievability": self.retrievability,
            "original_score": self.original_score,
            "fsrs_adjusted_score": self.fsrs_adjusted_score,
        }


def fsrs_adjust(
    chunk_id: str,
    score: float,
    *,
    stability: float,
    difficulty: float,
    elapsed_days: float,
    w20: float = DEFAULT_W20,
) -> FSRSWeightedResult:
    """Compute the FSRS-6 adjusted score for a single chunk.

    adjusted = score * retrievability

    Args:
        chunk_id: for the result record.
        score: input score (RRF, rerank, decay-adjusted, etc.).
        stability: chunk's stability in days.
        difficulty: chunk's difficulty in [1, 10].
        elapsed_days: days since last review.
        w20: FSRS-6 decay exponent (default 0.9).

    Returns:
        FSRSWeightedResult with R, S, D, and adjusted score.
    """
    r = fsrs_retrievability(elapsed_days, stability, w20=w20)
    return FSRSWeightedResult(
        chunk_id=chunk_id,
        elapsed_days=elapsed_days,
        stability_days=stability,
        difficulty=difficulty,
        retrievability=r,
        original_score=score,
        fsrs_adjusted_score=score * r,
    )


def days_since(epoch: float | int | None) -> float:
    """Convert a unix timestamp to days since then. Returns inf if epoch is None."""
    if epoch is None:
        return float("inf")
    now = time.time()
    return max(0.0, (now - float(epoch)) / 86400.0)


# -----------------------------------------------------------------------------
# Opt-in dispatch (matches the pattern in src/rerank.py and src/decay.py)
# -----------------------------------------------------------------------------


def _env_enabled() -> bool:
    return os.environ.get("DUCKBOT_FSRS", "").strip().lower() in ("1", "true", "yes", "on")


def maybe_fsrs(
    results: list[Any],
    *,
    enabled: bool | None = None,
    w20: float = DEFAULT_W20,
    w8: float = DEFAULT_W8,
    w6: float = DEFAULT_W6,
) -> list[Any]:
    """Apply FSRS-6 retrieval weighting to a list of results.

    Like src/decay.py's maybe_decay(), this is opt-in. Off by default
    (L8's simpler Ebbinghaus math is the default). When enabled, each
    result's `rrf_score` is multiplied by its retrievability R, and
    audit fields (`_fsrs_retrievability`, `_fsrs_stability`,
    `_fsrs_difficulty`) are attached.

    Args:
        results: list of objects with `.chunk_id`, `.tier`, `.rrf_score`,
            and `.metadata` attrs (typically `QueryResult` from
            src/query.py). Each result's metadata dict should ideally
            have `stability_days` and `difficulty`; otherwise defaults
            (7.0 stability, 5.0 difficulty) are used.
        enabled: True to force-on, False to force-off, None to honor
            DUCKBOT_FSRS env var.
        w20: FSRS-6 decay exponent (default 0.9).
        w8: FSRS-6 stability growth weight (default 0.02).
        w6: FSRS-6 difficulty update weight (default 0.1).

    Returns:
        New list sorted by adjusted score desc, or input unchanged
        if disabled.
    """
    if enabled is False:
        return results
    if enabled is None and not _env_enabled():
        return results
    if not results:
        return results

    annotated = []
    now = time.time()
    for r in results:
        meta = getattr(r, "metadata", None) or {}
        stability = float(meta.get("stability_days") or 7.0)
        difficulty = float(meta.get("difficulty") or DEFAULT_DIFFICULTY)

        # elapsed_days: prefer last_recalled_at; fall back to created_at;
        # fall back to ingested_at; if none, treat as "never reviewed"
        # (age = 0 → R = 1.0).
        last_seen = (
            meta.get("last_recalled_at")
            or meta.get("created_at")
            or meta.get("ingested_at")
        )
        if last_seen is None:
            elapsed = 0.0
        else:
            elapsed = max(0.0, (now - float(last_seen)) / 86400.0)

        r_now = fsrs_retrievability(elapsed, stability, w20=w20)
        original_rrf = getattr(r, "rrf_score", 0.0) or 0.0
        adjusted = original_rrf * r_now

        setattr(r, "_fsrs_retrievability", r_now)
        setattr(r, "_fsrs_stability", stability)
        setattr(r, "_fsrs_difficulty", difficulty)
        setattr(r, "_fsrs_elapsed_days", elapsed)
        setattr(r, "rrf_score", adjusted)
        annotated.append(r)

    annotated.sort(key=lambda r: r.rrf_score, reverse=True)
    return annotated


__all__ = [
    "DEFAULT_W20",
    "DEFAULT_W8",
    "DEFAULT_W6",
    "DEFAULT_DIFFICULTY",
    "DIFFICULTY_MIN",
    "DIFFICULTY_MAX",
    "STABILITY_FLOOR",
    "STABILITY_CEILING",
    "fsrs_retrievability",
    "fsrs_bump_stability",
    "fsrs_bump_difficulty",
    "fsrs_adjust",
    "days_since",
    "maybe_fsrs",
]
