"""
connectors/active_memory.py — alias layer for OpenClaw's Active Memory plugin.

OpenClaw's Active Memory plugin exposes tools under names like:
    - memory_query(query, k, tier_filter)
    - memory_store(text, source, tier, metadata)
    - memory_recent(k, tier)
    - memory_forget(query, k)

We accept those exact tool names and route them to the Brain facade so
the brain speaks the same protocol OpenClaw's other agents do.

This is NOT a re-implementation; it's a translation layer. The brain
keeps all its own semantics (RRF, decay, rerank, FSRS) and the alias
layer just maps names.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..memory import Memory
from .base import Brain


@dataclass
class ActiveMemoryResult:
    ok: bool = True
    tool: str = ""
    data: dict = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "tool": self.tool,
            "data": self.data,
            "error": self.error,
        }


class ActiveMemoryAdapter:
    """Translate OpenClaw's Active Memory tool names -> Brain methods."""

    def __init__(self, brain: Brain):
        self.brain = brain

    def memory_query(
        self,
        query: str,
        k: int = 5,
        tier_filter: Optional[str] = None,
    ) -> dict:
        r = self.brain.recall(query=query, k=k, tier=tier_filter, rerank=True, tier_priors=True)
        return ActiveMemoryResult(
            tool="memory_query",
            data={
                "query": query,
                "k": k,
                "tier_filter": tier_filter,
                "results": [
                    {
                        "chunk_id": x.chunk_id,
                        "text": x.text,
                        "tier": x.tier.value if hasattr(x.tier, "value") else str(x.tier),
                        "score": getattr(x, "score", 0.0),
                        "source_path": (x.metadata or {}).get("source_path", ""),
                    }
                    for x in r
                ],
            },
        ).to_dict()

    def memory_store(
        self,
        text: str,
        source: str = "<active_memory>",
        tier: str = "semantic",
        metadata: Optional[dict] = None,
    ) -> dict:
        # Use remember() with force_tier as a string. The v0.10.1 string
        # coercion fix handles that. remember() returns a RememberResult;
        # we surface its chunk_id (plus tier/importance for diagnostics).
        # A quarantined result has chunk_id=None — distinguish that from
        # a real store by also surfacing `quarantined`.
        r = self.brain.remember(
            text=text,
            source_path=source,
            force_tier=tier,
            metadata=metadata or {},
        )
        quarantined = bool(getattr(r, "quarantined", False))
        stored = bool(getattr(r, "stored", True)) and not quarantined
        return ActiveMemoryResult(
            tool="memory_store",
            data={
                "chunk_id": getattr(r, "chunk_id", None),
                "tier": getattr(r, "tier", None) or tier,
                "source": source,
                "stored": stored,
                "quarantined": quarantined,
            },
        ).to_dict()

    def memory_recent(self, k: int = 10, tier: Optional[str] = None) -> dict:
        s = self.brain.stats(include_vector_store=False)
        # Without a dedicated "recent" method, we sample recall() with no
        # query — this returns the most recent-ish chunks. Good enough.
        r = self.brain.recall(query="", k=k, tier=tier, rerank=False)
        return ActiveMemoryResult(
            tool="memory_recent",
            data={
                "k": k,
                "tier": tier,
                "stats": s.to_dict(),
                "results": [
                    {
                        "chunk_id": x.chunk_id,
                        "text": x.text,
                        "tier": x.tier.value if hasattr(x.tier, "value") else str(x.tier),
                        "source_path": (x.metadata or {}).get("source_path", ""),
                    }
                    for x in r
                ],
            },
        ).to_dict()

    def memory_forget(self, query: str, k: int = 5) -> dict:
        removed = self.brain.forget_by_query(query=query, k=k)
        return ActiveMemoryResult(
            tool="memory_forget",
            data={"query": query, "removed": removed, "k": k},
        ).to_dict()

    def call(self, tool: str, args: dict) -> dict:
        """Dispatch by tool name. Used by MCP-style callers."""
        method = getattr(self, tool, None)
        if method is None:
            return ActiveMemoryResult(
                ok=False,
                tool=tool,
                error=f"unknown active-memory tool: {tool}",
            ).to_dict()
        try:
            return method(**args)
        except TypeError as e:
            return ActiveMemoryResult(
                ok=False,
                tool=tool,
                error=f"bad args: {e}",
            ).to_dict()
        except Exception as e:
            return ActiveMemoryResult(
                ok=False,
                tool=tool,
                error=f"{type(e).__name__}: {e}",
            ).to_dict()


# -----------------------------------------------------------------------------
# Sync helper
# -----------------------------------------------------------------------------

def make_adapter(brain: Optional[Brain] = None) -> ActiveMemoryAdapter:
    if brain is None:
        brain = Brain()
    return ActiveMemoryAdapter(brain)
