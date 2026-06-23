import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tier import Tier, classify, classify_by_path, reclassify_for_working


def test_dated_log_is_episodic():
    from datetime import date, timedelta
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    a = classify(f"/Users/duckets/.openclaw/workspace/memory/{yesterday}.md", "Some content")
    assert a.tier == Tier.EPISODIC


def test_dated_log_with_suffix_is_episodic():
    from datetime import date, timedelta
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    a = classify(f"/Users/duckets/.openclaw/workspace/memory/{yesterday}-evening.md", "x")
    assert a.tier == Tier.EPISODIC


def test_agents_md_is_procedural():
    a = classify("/Users/duckets/.openclaw/workspace/AGENTS.md", "rules here")
    assert a.tier == Tier.PROCEDURAL


def test_soul_md_is_procedural():
    a = classify("/Users/duckets/.openclaw/workspace/SOUL.md", "voice rules")
    assert a.tier == Tier.PROCEDURAL


def test_memory_md_is_semantic():
    a = classify("/Users/duckets/.openclaw/workspace/MEMORY.md", "durable facts")
    assert a.tier == Tier.SEMANTIC


def test_readme_md_is_semantic():
    a = classify("/some/project/README.md", "project overview")
    assert a.tier == Tier.SEMANTIC


def test_today_md_is_working():
    """Today's daily log should be classified as WORKING tier."""
    from datetime import date
    today = date.today().isoformat()
    a = classify(f"/Users/duckets/.openclaw/workspace/memory/{today}.md", "active session")
    assert a.tier == Tier.WORKING


def test_unknown_path_falls_back_to_content():
    a = classify("/random/path/file.txt", "today we did some stuff at 5:30 PM")
    # Has temporal markers → episodic
    assert a.tier in (Tier.EPISODIC, Tier.WORKING)


def test_imperative_voice_is_procedural():
    a = classify("/unknown/file.md", "Always commit before pushing. Never skip tests.")
    assert a.tier == Tier.PROCEDURAL


def test_high_confidence_for_path_match():
    a = classify("/Users/duckets/.openclaw/workspace/AGENTS.md", "anything")
    assert a.confidence >= 0.9


def test_low_confidence_for_content_only():
    a = classify("/unknown.md", "random text without markers")
    # Falls back to default episodic with low confidence
    assert a.confidence <= 0.6


def test_reclassify_working_only_for_today():
    from datetime import date, timedelta
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    a = classify(f"/Users/duckets/.openclaw/workspace/memory/{yesterday}.md", "x")
    promoted = reclassify_for_working(a.source_path, a)
    # Yesterday's log is not today → no promotion
    assert promoted.tier == a.tier
    assert promoted.tier != Tier.WORKING


def test_tier_enum_values():
    assert Tier.WORKING.value == "working"
    assert Tier.EPISODIC.value == "episodic"
    assert Tier.SEMANTIC.value == "semantic"
    assert Tier.PROCEDURAL.value == "procedural"


def test_classify_by_path_returns_none_for_unknown():
    result = classify_by_path("/totally/unknown/path.md")
    assert result is None