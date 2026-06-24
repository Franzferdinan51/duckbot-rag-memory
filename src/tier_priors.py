"""
tier_priors.py — Layer 11: weighted RRF with per-tier priors.

Standard RRF (Cormack et al. 2009, used in src/query.py) treats every
chunk equally once their fused rank is computed. That's a problem when
your corpus mixes a "never push secrets" rule (procedural) with a
2026-06-23 chat log (episodic) — the chat log can outrank the rule if
it has more keyword matches in BM25.

Pattern from Cognee's "tier-aware RRF" (Apache-2.0) and MemPalace's
per-section weight map (MIT): apply a multiplicative prior to each
hit's RRF contribution based on its tier.

  adjusted_score(d) = prior(tier(d)) * rrf_score(d)

Default priors (designed for our actual brain hierarchy):
  procedural: 1.50  (rules / behavioral norms — should win ties)
  semantic:   1.20  (curated long-term facts)
  episodic:   1.00  (baseline — no boost)
  working:    0.80  (today's chatter — shouldn't outrank rules)

All priors are clamped to [0.1, 3.0] to prevent pathological values
from destroying ranking sanity. Overridable per-call via the `priors`
dict argument.

Opt-in. Default OFF. Three ways to enable:
  - DUCKBOT_TIER_PRIORS=1
  - tier_priors=True kwarg to hybrid_query()
  - tier_priors_prior_overrides dict for custom weights

When disabled, the function is a no-op (returns results unchanged) so
callers that don't opt in keep the standard RRF behavior.
"""

from __future__ import annotations

import os
from typing import Any

from .tier import Tier


# -----------------------------------------------------------------------------
# Default priors
# -----------------------------------------------------------------------------

DEFAULT_PRIORS: dict[str, float] = {
    "procedural": 1.50,
    "semantic": 1.20,
    "episodic": 1.00,
    "working": 0.80,
}

# Clamp bounds — don't let a misconfigured prior total-break ranking.
PRIOR_MIN = 0.1
PRIOR_MAX = 3.0


def get_prior(tier: str, overrides: dict[str, float] | None = None) -> float:
    """Return the prior weight for a tier. Clamped to [PRIOR_MIN, PRIOR_MAX]."""
    prior = (overrides or {}).get(tier, DEFAULT_PRIORS.get(tier, 1.0))
    return max(PRIOR_MIN, min(PRIOR_MAX, prior))


# -----------------------------------------------------------------------------
# Opt-in dispatch (matches the rerank.py / decay.py pattern)
# -----------------------------------------------------------------------------


def _env_enabled() -> bool:
    """True if DUCKBOT_TIER_PRIORS=1 is set."""
    return os.environ.get("DUCKBOT_TIER_PRIORS", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def maybe_apply_tier_priors(
    results: list[Any],
    enabled: bool | None = None,
    overrides: dict[str, float] | None = None,
) -> list[Any]:
    """Apply tier priors to a list of QueryResult-like objects.

    Args:
        results: List of objects with `.tier` and `.rrf_score` attrs
            (typically `QueryResult` from src/query.py).
        enabled: True to force-on, False to force-off, None to honor
            DUCKBOT_TIER_PRIORS env var.
        overrides: Optional dict mapping tier name -> prior weight.
            Tier names not in the dict fall back to DEFAULT_PRIORS.

    Returns:
        New list sorted by adjusted score desc. Original list is not
        mutated. If disabled, returns the input list unchanged (so
        callers that don't opt in keep the standard RRF behavior).

    The function is best-effort: chunks with missing tier fall back
    to prior 1.0 (episodic baseline). Tier names that aren't in our
    enum also get 1.0.
    """
    if enabled is False:
        return results
    if enabled is None and not _env_enabled():
        return results

    if not results:
        return results

    # Apply the prior, attach an audit field, sort.
    annotated = []
    for r in results:
        tier = getattr(r, "tier", None) or "episodic"
        prior = get_prior(tier, overrides)
        original_rrf = getattr(r, "rrf_score", 0.0) or 0.0
        adjusted = prior * original_rrf
        # Always set the audit fields + adjusted score.
        setattr(r, "_tier_prior", prior)
        setattr(r, "_rrf_score_pre_prior", original_rrf)
        setattr(r, "rrf_score", adjusted)
        annotated.append(r)

    annotated.sort(key=lambda r: r.rrf_score, reverse=True)
    return annotated


__all__ = [
    "DEFAULT_PRIORS",
    "PRIOR_MIN",
    "PRIOR_MAX",
    "get_prior",
    "maybe_apply_tier_priors",
]
