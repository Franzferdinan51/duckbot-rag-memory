"""
fsrs_optimizer.py — self-tune FSRS w20 from recall history.

Background: the FSRS-6 spec ships 17 weights tuned on Anki users. The
key one for us is w20 (the forgetting-curve exponent). Different
deployments have different w20: a daily-flashcard app has w20 ~ 0.5
(steeper forgetting); a long-form knowledge base like ours wants a
flatter curve (w20 ~ 1.0-1.5) so dense technical notes don't decay
in days.

Without labeled recall data, we can't run the full RSME optimizer
from the FSRS paper. We CAN use a much cheaper proxy:

  For each chunk in the brain:
    - t = max(0, now - last_recalled_at)  (days since last recall)
    - if recall_count == 0: label = 0 (forgotten — never retrieved)
    - else:                label = 1 (remembered — actively used)
    - predicted_R = (1 + t / (9 * S)) ^ (-w20)

  Loss: sum of (predicted_R - label)^2 over all chunks (MSE)
  Search: grid search over w20 ∈ [0.05, 3.0] in 0.05 steps.

Why this works: a w20 that's too HIGH makes the curve steep — old
chunks look "forgotten" even though recall_count shows they were
remembered. A w20 that's too LOW makes the curve flat — even
forgotten chunks look "fresh". The MSE-optimal w20 best matches the
actual recall pattern.

This is a v0.1 approximation. For full FSRS optimization we'd want
the user's review history (right/wrong on each review) — see
https://github.com/open-spaced-repetition/py-fsrs/tree/master/fsrs/optimizer

Cost: zero (pure Python, no LLM call). Runs in <1 second on a 5000-chunk
brain.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class FitResult:
    """Result of fitting w20 to recall history."""
    best_w20: float
    best_mse: float
    n_chunks: int
    n_remembered: int
    n_forgotten: int
    # History of (w20, mse) for plotting / inspection.
    sweep: list[tuple[float, float]]
    # The default w20 we compare against.
    baseline_w20: float = 0.9
    baseline_mse: float = 0.0

    def to_dict(self) -> dict:
        return {
            "best_w20": self.best_w20,
            "best_mse": self.best_mse,
            "n_chunks": self.n_chunks,
            "n_remembered": self.n_remembered,
            "n_forgotten": self.n_forgotten,
            "baseline_w20": self.baseline_w20,
            "baseline_mse": self.baseline_mse,
            "improvement_pct": round(
                (self.baseline_mse - self.best_mse) / max(self.baseline_mse, 1e-9) * 100, 2
            ),
        }


def _retrievability(t_days: float, stability: float, w20: float) -> float:
    """FSRS-6 retrievability. Mirrors fsrs.fsrs_retrievability to avoid
    an import cycle if this module is used standalone."""
    if stability <= 0 or w20 <= 0:
        return 0.0
    if t_days < 0:
        t_days = 0.0
    try:
        r = (1.0 + t_days / (9.0 * stability)) ** (-w20)
    except (OverflowError, ValueError):
        r = 0.0
    return max(0.0, min(1.0, r))


def fit_w20(
    chunks: list[dict],
    *,
    default_w20: float = 0.9,
    w20_lo: float = 0.05,
    w20_hi: float = 3.0,
    w20_step: float = 0.05,
    now: Optional[float] = None,
) -> FitResult:
    """Find the w20 that best matches the recall pattern.

    Args:
        chunks: list of dicts with keys
            - "stability_days" (float, S in FSRS-6)
            - "last_recalled_at" (float epoch, 0 = never)
            - "recall_count" (int, 0+)
        default_w20: comparison baseline (typically our shipped 0.9).
        w20_lo, w20_hi, w20_step: search grid.
        now: current time (default time.time()).

    Returns:
        FitResult with best w20 + full sweep.
    """
    if not chunks:
        return FitResult(
            best_w20=default_w20, best_mse=0.0,
            n_chunks=0, n_remembered=0, n_forgotten=0, sweep=[],
        )
    if now is None:
        now = time.time()

    # Build (t, label, S) tuples. Label = 1 if recalled at least once.
    samples: list[tuple[float, int, float]] = []
    for c in chunks:
        try:
            S = float(c.get("stability_days") or 0.0)
        except (TypeError, ValueError):
            S = 0.0
        if S <= 0:
            continue  # can't predict retrievability without stability
        last_recall = float(c.get("last_recalled_at") or 0.0)
        rc = int(c.get("recall_count") or 0)
        if last_recall <= 0:
            # Never recalled — treat as "for a long time" (full decay)
            t = (now - float(c.get("ingested_at") or now)) / 86400.0
        else:
            t = (now - last_recall) / 86400.0
        t = max(0.0, t)
        label = 1 if rc > 0 else 0
        samples.append((t, label, S))

    if not samples:
        return FitResult(
            best_w20=default_w20, best_mse=0.0,
            n_chunks=len(chunks), n_remembered=0, n_forgotten=0, sweep=[],
        )

    n_rem = sum(1 for _, lbl, _ in samples if lbl == 1)
    n_for = sum(1 for _, lbl, _ in samples if lbl == 0)

    # Grid search: minimize sum of (pred - label)^2.
    sweep: list[tuple[float, float]] = []
    best_w, best_mse = default_w20, float("inf")
    w = w20_lo
    while w <= w20_hi + 1e-9:
        sse = 0.0
        for t, lbl, S in samples:
            pred = _retrievability(t, S, w)
            sse += (pred - lbl) ** 2
        mse = sse / len(samples)
        sweep.append((round(w, 4), round(mse, 6)))
        if mse < best_mse:
            best_mse = mse
            best_w = w
        w += w20_step

    # Baseline MSE at the default w20.
    sse_base = 0.0
    for t, lbl, S in samples:
        pred = _retrievability(t, S, default_w20)
        sse_base += (pred - lbl) ** 2
    base_mse = sse_base / len(samples)

    return FitResult(
        best_w20=round(best_w, 4),
        best_mse=round(best_mse, 6),
        n_chunks=len(samples),
        n_remembered=n_rem,
        n_forgotten=n_for,
        sweep=sweep,
        baseline_w20=default_w20,
        baseline_mse=round(base_mse, 6),
    )


def fit_and_apply(
    store,
    *,
    default_w20: float = 0.9,
    w20_lo: float = 0.05,
    w20_hi: float = 3.0,
    w20_step: float = 0.05,
) -> FitResult:
    """Fit w20 from the live store and apply it. Convenience wrapper
    that pulls chunks, runs fit_w20, and returns the result. The brain
    doesn't auto-apply — call this from a cron or brain_optimize_fsrs
    MCP tool, and the caller decides whether to commit.

    Returns:
        FitResult. `to_dict()` includes the proposed w20 + improvement
        vs baseline so the caller can decide.
    """
    # Pull chunks from the store.
    from .tier import Tier
    chunks: list[dict] = []
    for tier_name in ("working", "episodic", "semantic", "procedural"):
        try:
            coll = store.collection_for(Tier(tier_name))
            data = coll.get(include=["metadatas"], limit=10000)
        except Exception:
            continue
        metas = (data or {}).get("metadatas") or []
        for md in metas:
            md = md or {}
            if md.get("superseded_by"):
                continue
            chunks.append({
                "stability_days": md.get("fsrs_stability_days") or md.get("stability_days"),
                "last_recalled_at": md.get("last_recalled_at") or md.get("fsrs_last_review_ts"),
                "recall_count": md.get("recall_count") or 0,
                "ingested_at": md.get("ingested_at") or 0,
            })
    return fit_w20(
        chunks,
        default_w20=default_w20,
        w20_lo=w20_lo, w20_hi=w20_hi, w20_step=w20_step,
    )
