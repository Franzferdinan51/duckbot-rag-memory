"""
ratelimit.py — per-tool token-bucket rate limiter for the MCP server.

Defends the brain against misbehaving agents that spam a single tool
(e.g. brain_remember in a tight loop). Each tool has its own bucket;
once exhausted, the MCP handler returns a 429-style error.

Default limits (per minute):
  - brain_wake_up:        60   (cheap, called once per session)
  - brain_recall:         60
  - brain_remember:       10   (expensive: ingests + embeds)
  - brain_inflate:        30
  - brain_sync:           10
  - brain_nudge:          30
  - brain_palace:         30
  - brain_seed_demo:      10
  - everything else:      60

Disabled by setting DUCKBOT_RATELIMIT_DISABLE=1.

The buckets are per-process (not per-user) since the MCP server is a
single-tenant design — one operator, one server. If multi-tenancy is
ever added, the bucket map can be moved into a per-connection dict.
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from threading import Lock


# Per-tool limits (per minute). Tools not listed get the default below.
LIMITS_PER_MIN: dict[str, int] = {
    "brain_remember":    10,
    "brain_sync":        10,
    "brain_seed_demo":   10,
    "brain_inflate":     30,
    "brain_nudge":       30,
    "brain_palace":      30,
    "brain_wake_up":     60,
    "brain_recall":      60,
    "recall":            60,
    "remember":          10,
    "reflect":           10,
}

DEFAULT_LIMIT_PER_MIN = 60


class _Bucket:
    __slots__ = ("tokens", "last_refill")

    def __init__(self, tokens: float, last_refill: float):
        self.tokens = tokens
        self.last_refill = last_refill


class RateLimiter:
    """Process-wide token-bucket rate limiter.

    Each tool has its own bucket; one bucket is consumed per call.
    Refill rate = (limit / 60) tokens per second.
    """

    def __init__(self):
        # Tool-name -> Bucket. Lazy creation via defaultdict so the bucket
        # starts at the per-tool limit on first call (we don't know the
        # tool name until check() is called).
        self._buckets: dict[str, _Bucket] = {}
        self._lock = Lock()

    def _new_bucket(self, tool_name: str) -> _Bucket:
        # Start full so the first call never blocks.
        return _Bucket(
            tokens=float(self._limit_for(tool_name)),
            last_refill=time.time(),
        )

    def is_disabled(self) -> bool:
        return os.environ.get("DUCKBOT_RATELIMIT_DISABLE", "0").lower() in (
            "1", "true", "yes",
        )

    def _limit_for(self, tool_name: str) -> int:
        return LIMITS_PER_MIN.get(tool_name, DEFAULT_LIMIT_PER_MIN)

    def check(self, tool_name: str) -> tuple[bool, dict]:
        """Try to consume one token from the tool's bucket.

        Returns (allowed, info) where info is a dict with retry_after
        (seconds) and current tokens (for diagnostics). Always
        allowed when is_disabled() returns True.
        """
        if self.is_disabled():
            return True, {"disabled": True, "current_tokens": float("inf")}
        with self._lock:
            limit = self._limit_for(tool_name)
            # Compute refill based on elapsed seconds.
            now = time.time()
            b = self._buckets.get(tool_name)
            if b is None:
                b = self._new_bucket(tool_name)
                self._buckets[tool_name] = b
            elapsed = now - b.last_refill
            if elapsed > 0:
                refill = (limit / 60.0) * elapsed
                b.tokens = min(float(limit), b.tokens + refill)
                b.last_refill = now
            if b.tokens >= 1.0:
                b.tokens -= 1.0
                return True, {
                    "limit_per_min": limit,
                    "current_tokens": b.tokens,
                    "retry_after": 0.0,
                }
            # Exhausted: time until the next token = (1 - tokens) / refill_rate
            rate = limit / 60.0
            retry_after = (1.0 - b.tokens) / rate if rate > 0 else 60.0
            return False, {
                "limit_per_min": limit,
                "current_tokens": b.tokens,
                "retry_after": round(retry_after, 3),
            }


# Module-level singleton.
_RATE_LIMITER: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _RATE_LIMITER
    if _RATE_LIMITER is None:
        _RATE_LIMITER = RateLimiter()
    return _RATE_LIMITER


def reset_rate_limiter() -> None:
    """Test helper: drop the cached RateLimiter so the next test
    starts with a fresh bucket map."""
    global _RATE_LIMITER
    _RATE_LIMITER = None
