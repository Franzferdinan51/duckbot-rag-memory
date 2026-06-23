import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.consolidate import extract_facts_from_chunk, deduplicate_facts, ExtractedFact


def test_extract_user_said():
    text = "Duckets said 'I always want rich interactive messages.'"
    facts = extract_facts_from_chunk(text, "id1", "/fake/memory.md")
    assert any(f.kind == "user-said" for f in facts)


def test_extract_decision():
    text = "We decided to use cloud-only models for all agent work."
    facts = extract_facts_from_chunk(text, "id1", "/fake.md")
    assert any(f.kind == "decision" for f in facts)


def test_extract_rule():
    text = "Always commit before pushing to origin. Never skip tests."
    facts = extract_facts_from_chunk(text, "id1", "/fake.md")
    rule_facts = [f for f in facts if f.kind == "rule"]
    assert len(rule_facts) >= 1


def test_extract_setup():
    text = "Installed cua-driver at ~/.local/bin/cua-driver"
    facts = extract_facts_from_chunk(text, "id1", "/fake.md")
    assert any(f.kind == "setup" for f in facts)


def test_extract_location():
    text = "Duckets' home address: 7516 Pomeranian Drive, Huber Heights, OH 45424."
    facts = extract_facts_from_chunk(text, "id1", "/fake.md")
    assert any(f.kind == "location" for f in facts)


def test_extract_strips_markdown():
    text = "Duckets said 'the project uses **chromadb** for vector storage.'"
    facts = extract_facts_from_chunk(text, "id1", "/fake.md")
    rule_facts = [f for f in facts if f.kind == "user-said"]
    assert any("**" not in f.text for f in rule_facts)


def test_dedup_identical_facts():
    facts = [
        ExtractedFact(text="cloud-only", kind="rule", source_chunk_id="1", source_path="x"),
        ExtractedFact(text="cloud-only", kind="rule", source_chunk_id="2", source_path="y"),
    ]
    deduped = deduplicate_facts(facts)
    assert len(deduped) == 1


def test_dedup_keeps_higher_confidence():
    # Two near-identical facts (same words, one adds tiny detail) — should dedup
    f1 = ExtractedFact(text="cloud-only models", kind="rule", source_chunk_id="1", source_path="x", confidence=0.4)
    f2 = ExtractedFact(text="cloud-only models now", kind="rule", source_chunk_id="2", source_path="y", confidence=0.9)
    deduped = deduplicate_facts([f1, f2])
    assert len(deduped) == 1
    assert deduped[0].confidence == 0.9


def test_dedup_keeps_distinct_facts():
    facts = [
        ExtractedFact(text="cloud-only models", kind="rule", source_chunk_id="1", source_path="x"),
        ExtractedFact(text="local home address is in Huber Heights", kind="location", source_chunk_id="2", source_path="y"),
    ]
    deduped = deduplicate_facts(facts)
    assert len(deduped) == 2


def test_empty_input():
    facts = extract_facts_from_chunk("", "id", "x.md")
    assert facts == []


def test_short_facts_rejected():
    text = "x said y."  # too short
    facts = extract_facts_from_chunk(text, "id", "x.md")
    assert all(len(f.text) >= 5 for f in facts)


def test_facts_have_source_provenance():
    text = "Duckets said use cloud-only models"
    facts = extract_facts_from_chunk(text, "chunk-42", "/path/to/file.md")
    for f in facts:
        assert f.source_chunk_id == "chunk-42"
        assert f.source_path == "/path/to/file.md"