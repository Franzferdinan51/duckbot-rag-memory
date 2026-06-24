"""
test_entities.py — tests for entity extraction (Layer 2).
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph import Graph
from src.entities import EntityExtractor, ExtractedEntity, ExtractedTriple, extract_from_markdown_file


@pytest.fixture
def extractor():
    return EntityExtractor()


@pytest.fixture
def graph(tmp_path):
    g = Graph(path=tmp_path / "graph.db")
    yield g
    g.close()


# --- Entity recognition ----------------------------------------------------

def test_extracts_person_by_name(extractor):
    ents, _ = extractor.extract("Duckets said the new build is great.")
    names = [e.name for e in ents]
    assert "Duckets" in names
    assert any(e.kind == "person" for e in ents if e.name == "Duckets")


def test_extracts_known_project(extractor):
    text = "OpenClaw is the gateway, and DuckBot runs on top of it."
    ents, _ = extractor.extract(text)
    names = {e.name for e in ents}
    assert "OpenClaw" in names
    assert "DuckBot" in names


def test_extracts_file_paths(extractor):
    text = "Edit src/cli.py and tests/test_query.py to add the new feature."
    ents, _ = extractor.extract(text)
    names = {e.name for e in ents}
    assert "src/cli.py" in names, f"got names: {names}"
    assert "tests/test_query.py" in names, f"got names: {names}"


def test_extracts_openclaw_paths(extractor):
    text = "Config lives at ~/.openclaw/openclaw.json and the workspace at ~/.openclaw/workspace/memory/"
    ents, _ = extractor.extract(text)
    names = {e.name for e in ents}
    assert "~/.openclaw/openclaw.json" in names


def test_extracts_birthday_fact(extractor):
    text = "Duckets birthday: April 20th, can't do surprises that day."
    ents, _ = extractor.extract(text)
    facts = [e for e in ents if e.kind == "fact"]
    assert any("Duckets birthday" == e.name for e in facts)


def test_skips_pronouns(extractor):
    text = "He said he would check it later."
    ents, _ = extractor.extract(text)
    names = [e.name for e in ents]
    assert "He" not in names


# --- Relationship extraction ------------------------------------------------

def test_extracts_works_on_relationship(extractor):
    text = "Duckets works on OpenClaw. Duckets maintains DuckBot."
    _, triples = extractor.extract(text)
    labels = [(t.source, t.target, t.label) for t in triples]
    assert ("Duckets", "OpenClaw", "works_on") in labels, f"got {labels}"
    assert ("Duckets", "DuckBot", "works_on") in labels, f"got {labels}"


def test_extracts_uses_relationship(extractor):
    text = "OpenClaw uses MiniMax for cloud inference."
    _, triples = extractor.extract(text)
    labels = [t.label for t in triples]
    assert "uses" in labels, f"got {labels}"


def test_extracts_rotated_relationship(extractor):
    text = "Duckets rotated the Telegram bot token. Duckets also rotated the Tavily key."
    _, triples = extractor.extract(text)
    labels = [t.label for t in triples]
    assert "rotated" in labels, f"got {labels}"


def test_extracts_replaced_relationship(extractor):
    # Active voice: regex can catch this
    text = "Duckets replaced the old token with a new one."
    _, triples = extractor.extract(text)
    labels = [t.label for t in triples]
    assert "replaced" in labels, f"got {labels}"


def test_filters_junk_subjects(extractor):
    """Conjunctions and articles should not be used as subjects."""
    text = "And also we should think about it."
    _, triples = extractor.extract(text)
    for t in triples:
        assert t.source.lower() not in ("and", "or", "but", "also", "then", "so", "he", "she", "it")


# --- Cognify (load into graph) ---------------------------------------------

def test_cognify_loads_entities_and_triples(extractor, graph):
    text = "Duckets works on OpenClaw. OpenClaw uses MiniMax."
    ents, trips = extractor.extract(text)
    stats = extractor.cognify(graph, ents, trips, source="test.md")
    assert stats["entities_added"] >= 2
    assert stats["triples_added"] >= 2
    # Verify in graph
    duckets = graph.find_entity("Duckets")
    assert duckets is not None
    openclaw = graph.find_entity("OpenClaw")
    assert openclaw is not None
    rels = graph.query_active(entity_id=duckets.id, label="works_on")
    assert len(rels) == 1


def test_cognify_is_idempotent(extractor, graph):
    text = "Duckets works on OpenClaw."
    ents, trips = extractor.extract(text)
    s1 = extractor.cognify(graph, ents, trips, source="test.md")
    s2 = extractor.cognify(graph, ents, trips, source="test.md")
    # Re-running should not re-add entities
    assert s2["entities_added"] == 0, f"got {s2}"
    # Re-running should not create new active relationships in the graph
    active = graph.query_active(label="works_on")
    assert len(active) == 1, f"graph should have exactly 1 active works_on, got {len(active)}"


def test_cognify_auto_creates_unknown_endpoints(extractor, graph):
    """If a triple references an entity not in the entity list, cognify
    should auto-create it as a concept."""
    ents = []  # no entities
    trips = [ExtractedTriple(source="Foo", target="Bar", label="knows")]
    stats = extractor.cognify(graph, ents, trips, source="test.md")
    assert stats["entities_added"] == 2
    assert graph.find_entity("Foo") is not None
    assert graph.find_entity("Bar") is not None


# --- Batch extract ---------------------------------------------------------

def test_extract_batch_dedupes(extractor):
    texts = [
        "Duckets works on OpenClaw.",
        "OpenClaw uses MiniMax.",
        "Duckets maintains OpenClaw.",
    ]
    ents, trips = extractor.extract_batch(texts)
    name_list = [e.name for e in ents]
    # Should not have duplicate Duckets/OpenClaw
    assert name_list.count("Duckets") == 1, f"got {name_list}"
    assert name_list.count("OpenClaw") == 1, f"got {name_list}"
    # Should have multiple triples
    assert len(trips) >= 2


# --- LLM extractor (optional) ----------------------------------------------

def test_llm_extractor_failure_is_non_fatal(graph):
    """If the LLM extractor raises, regex results should still be returned."""

    class BoomExtractor:
        def __call__(self, texts):
            raise RuntimeError("LLM unavailable")

    ext = EntityExtractor(llm_extractor=BoomExtractor())
    text = "Duckets works on OpenClaw."
    ents, trips = ext.extract(text)
    # Regex still works
    assert any(e.name == "Duckets" for e in ents)
    assert any(t.label == "works_on" for t in trips)


def test_llm_extractor_adds_extra_triples(graph):
    class FakeLLM:
        def __call__(self, texts):
            return [[ExtractedTriple(source="Duckets", target="masterdashboard", label="owns")]]

    ext = EntityExtractor(llm_extractor=FakeLLM())
    text = "Random prose about OpenClaw."
    _, trips = ext.extract(text)
    labels = [(t.source, t.target, t.label) for t in trips]
    assert ("Duckets", "masterdashboard", "owns") in labels


# --- Integration with file -------------------------------------------------

def test_extract_from_real_memory_file(tmp_path):
    """Sanity test against an actual memory log."""
    md = tmp_path / "test.md"
    md.write_text("""# Daily Log

Duckets rotated the Telegram bot token. Duckets also rotated the Tavily key.
OpenClaw uses MiniMax.
OpenClaw depends on MiniMax for cloud inference.
""")
    ents, trips = extract_from_markdown_file(str(md))
    names = {e.name for e in ents}
    assert "Duckets" in names
    assert "OpenClaw" in names
    assert "MiniMax" in names
    labels = [t.label for t in trips]
    # At least one of each kind should be found
    assert "rotated" in labels, f"got {labels}"
    assert "depends_on" in labels, f"got {labels}"
