"""tools.py - DuckBot brain tool definitions for Hermes MemoryProvider plugin."""
# MIT License - see LICENSE in the repository root.
from __future__ import annotations

TOOL_DEFINITIONS: list[dict] = [
    {
        "name": "brain_wake_up",
        "description": (
            "ONE-CALL session-startup context load. Returns recent memories "
            "(superseded filtered), active blocks, graph summary, FSRS review "
            "queue, and stats. Call this FIRST on every session start."
        ),
    },
    {
        "name": "brain_recall",
        "description": (
            "Hybrid retrieval (vector + BM25 + RRF). Optional: rerank=true "
            "for cross-encoder boost, decay=true for Ebbinghaus retention, "
            "tier_priors=true for per-tier multiplicative weighting (Layer 11), "
            "fsrs=true for FSRS-6 power-law forgetting (Layer 9)."
        ),
    },
    {
        "name": "brain_recall_verbatim",
        "description": (
            "Returns the original (pre-overlap, pre-prefix) source text -- "
            "never paraphrased. Use when the user asks what exactly did I say?."
        ),
    },
    {
        "name": "brain_remember",
        "description": (
            "Persist a memory. Non-blocking by default (returns status=queued). "
            "Pass kind=skill_candidate to stamp a lightweight procedural-tier "
            "chunk for the agent-driven skill pipeline (no LLM, returns chunk_id). "
            "Pass facts=[...] to store pre-extracted facts as semantic-tier chunks."
        ),
    },
    {
        "name": "brain_reflect",
        "description": (
            "Sleep-time consolidation: merge episodic chunks into the semantic tier. "
            "Long-running (seconds to minutes for large brains); call once per day via cron."
        ),
    },
    {
        "name": "brain_stats",
        "description": "One-glance snapshot: chunk counts, graph entities, blocks, quarantine.",
    },
    {
        "name": "brain_fsrs_review",
        "description": "Return chunks due for FSRS-6 spaced-repetition review (R < 0.9).",
    },
    {
        "name": "brain_decay_status",
        "description": "Return Ebbinghaus decay status (R = e^{-t/S}) for recent chunks.",
    },
    {
        "name": "brain_search_verbatim",
        "description": "Exact substring match against the verbatim (pre-overlap) text.",
    },
    {
        "name": "brain_skills_list",
        "description": (
            "List unpromoted skill-candidate chunks sorted by recency then importance. "
            "The AGENT decides which to promote -- brain does no LLM work."
        ),
    },
    {
        "name": "brain_skills_suggest",
        "description": (
            "Semantic top-N skill candidates matching a query. Hybrid retrieval "
            "scoped to procedural tier. No LLM."
        ),
    },
    {
        "name": "brain_skills_promote",
        "description": (
            "Promote a skill candidate to a full agentskills.io SKILL.md. "
            "The AGENT authors name/description/instructions -- brain is pure "
            "storage + template (no LLM). Writes skills/<slug>/SKILL.md.",
        ),
    },
]

TOOL_NAME_TO_INDEX: dict[str, dict] = {t["name"]: t for t in TOOL_DEFINITIONS}

def get_tool_definition(name: str) -> dict | None:
    return TOOL_NAME_TO_INDEX.get(name)

def get_tool_names() -> list[str]:
    return [t["name"] for t in TOOL_DEFINITIONS]

__all__ = ["TOOL_DEFINITIONS", "TOOL_NAME_TO_INDEX", "get_tool_definition", "get_tool_names"]
