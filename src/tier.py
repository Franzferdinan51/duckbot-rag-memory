"""
tier.py — CoALA-inspired memory tier classifier.

Maps chunks to one of four tiers based on source path + content heuristics.
Pulled from the CoALA paper (Princeton 2023, arxiv:2309.02427) which formalizes
working/episodic/semantic/procedural memory for LLM agents.

We also model this after:
  - mem0's hierarchical memory (user/agent/session)
  - Letta's tiered blocks (core/recall/archival)
  - Cognee's semantic graph layer
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class Tier(str, Enum):
    WORKING = "working"          # Today's active session, in-flight
    EPISODIC = "episodic"        # Dated events, session logs
    SEMANTIC = "semantic"        # Distilled facts, user prefs, entities
    PROCEDURAL = "procedural"    # Rules, behavioral norms, patterns

    @classmethod
    def _missing_(cls, value):
        """Coerce tier strings with surrounding whitespace or case noise."""
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized:
                for member in cls:
                    if member.value == normalized:
                        return member
        return None


def coerce_optional_tier(value: Tier | str | None) -> Tier | None:
    """Normalize an optional tier input.

    Blank strings become ``None`` so callers can treat whitespace-only input
    as "no tier filter" instead of raising a ValueError.
    """
    if value is None:
        return None
    if isinstance(value, Tier):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    return Tier(value)


# File patterns → tier. Order matters (first match wins).
PATH_RULES: list[tuple[re.Pattern[str], Tier]] = [
    # Procedural — explicit rule/style files
    (re.compile(r"/AGENTS\.md$", re.IGNORECASE), Tier.PROCEDURAL),
    (re.compile(r"/SOUL\.md$", re.IGNORECASE), Tier.PROCEDURAL),
    (re.compile(r"/CODE_OF_CONDUCT\.md$", re.IGNORECASE), Tier.PROCEDURAL),
    (re.compile(r"/CHANGELOG\.md$", re.IGNORECASE), Tier.PROCEDURAL),
    (re.compile(r"/README\.md$", re.IGNORECASE), Tier.SEMANTIC),  # project README → semantic
    # Episodic — dated memory flushes (YYYY-MM-DD.md pattern)
    (re.compile(r"/\d{4}-\d{2}-\d{2}(-[a-z0-9-]+)?\.md$", re.IGNORECASE), Tier.EPISODIC),
    (re.compile(r"/memory/\d{4}/\d{2}/", re.IGNORECASE), Tier.EPISODIC),
    # Semantic — the curated long-form memory
    (re.compile(r"/MEMORY\.md$", re.IGNORECASE), Tier.SEMANTIC),
    (re.compile(r"/memory/WORLD\.md$", re.IGNORECASE), Tier.SEMANTIC),
    (re.compile(r"/memory/PRIORITY_MAP\.md$", re.IGNORECASE), Tier.SEMANTIC),
]


# Content heuristics — fall back to these if no path rule matches.
CONTENT_RULES: list[tuple[re.Pattern[str], Tier, str]] = [
    # Procedural patterns: imperative voice, rules
    (
        re.compile(r"^\s*(?:never|always|must|should not|do not)\s+", re.IGNORECASE | re.MULTILINE),
        Tier.PROCEDURAL,
        "imperative voice",
    ),
    # Episodic patterns: dated entries, "today", "yesterday"
    (
        re.compile(r"\b(?:today|yesterday|this morning|this evening)\b", re.IGNORECASE),
        Tier.EPISODIC,
        "temporal marker",
    ),
    (
        re.compile(r"\b\d{1,2}:\d{2}\s*(?:AM|PM|EDT|EST|UTC)\b"),
        Tier.EPISODIC,
        "timestamp",
    ),
]


@dataclass(eq=False, frozen=True)
class TierAssignment:
    tier: Tier
    source_path: str
    rule_matched: str       # why we picked this tier
    confidence: float       # 0.0-1.0

    def __hash__(self) -> int:
        return hash((self.tier, self.source_path, self.rule_matched, self.confidence))


def _normalize_path(source_path: str) -> str:
    """Normalize Windows backslash to forward slash so PATH_RULES match on all OSes."""
    return source_path.replace("\\", "/")


def classify_by_path(source_path: str) -> TierAssignment | None:
    """Classify a chunk by its source file path. Returns None if no rule matches."""
    for pattern, tier in PATH_RULES:
        norm = _normalize_path(source_path)
        if pattern.search(norm):
            return TierAssignment(
                tier=tier,
                source_path=source_path,
                rule_matched=f"path: {pattern.pattern}",
                confidence=0.9,
            )
    return None


def classify_by_content(text: str) -> TierAssignment:
    """Classify by content heuristics. Used as a fallback when path has no rule."""
    matches: list[tuple[Tier, str]] = []
    for pattern, tier, why in CONTENT_RULES:
        if pattern.search(text):
            matches.append((tier, why))

    if not matches:
        # Default: most daily logs are episodic, so lean that way
        return TierAssignment(
            tier=Tier.EPISODIC,
            source_path="<unknown>",
            rule_matched="default (no match)",
            confidence=0.3,
        )

    # If we have multiple matches, prefer procedural > semantic > episodic > working.
    tier_priority = {
        Tier.PROCEDURAL: 4,
        Tier.SEMANTIC: 3,
        Tier.EPISODIC: 2,
        Tier.WORKING: 1,
    }
    matches.sort(key=lambda m: tier_priority[m[0]], reverse=True)
    return TierAssignment(
        tier=matches[0][0],
        source_path="<unknown>",
        rule_matched=f"content: {matches[0][1]}",
        confidence=0.5,
    )


def classify(source_path: str, text: str) -> TierAssignment:
    """Classify a chunk into one of the four CoALA tiers.

    Always runs reclassify_for_working last so today's dated log
    is automatically promoted from EPISODIC to WORKING.
    """
    by_path = classify_by_path(source_path)
    if by_path is not None:
        return reclassify_for_working(source_path, by_path)
    assignment = classify_by_content(text)
    # Re-create with correct source_path (frozen dataclass — no in-place mutation)
    assignment = TierAssignment(
        tier=assignment.tier,
        source_path=source_path,
        rule_matched=assignment.rule_matched,
        confidence=assignment.confidence,
    )
    return reclassify_for_working(source_path, assignment)


def is_working_tier_for_today(source_path: str) -> bool:
    """Working tier is for today's active session. Conventionally:
    `memory/today.md` or the most-recently-modified daily file.
    """
    from datetime import date

    name = Path(source_path).name
    return name in {f"{date.today().isoformat()}.md", "today.md", "current.md"}


# Re-classify a chunk as WORKING if it's from today's file.
def reclassify_for_working(source_path: str, current: TierAssignment) -> TierAssignment:
    if is_working_tier_for_today(source_path):
        return TierAssignment(
            tier=Tier.WORKING,
            source_path=source_path,
            rule_matched="today's active session",
            confidence=0.95,
        )
    return current
