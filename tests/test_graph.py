"""
test_graph.py — tests for the temporal knowledge graph (Layer 1).

These tests use tmp_path so they don't touch the real graph database.
"""

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graph import Graph, Entity, Relationship, DEFAULT_GRAPH_PATH


@pytest.fixture
def graph(tmp_path):
    db = tmp_path / "test-graph.db"
    g = Graph(path=db)
    yield g
    g.close()


def test_create_and_get_entity(graph):
    e = graph.upsert_entity("Duckets", "person", aliases=["Ryan", "duckets_mcquackin"])
    assert e.id
    assert e.name == "Duckets"
    assert "Ryan" in e.aliases
    fetched = graph.get_entity(e.id)
    assert fetched is not None
    assert fetched.name == "Duckets"


def test_upsert_entity_is_idempotent_by_name(graph):
    e1 = graph.upsert_entity("OpenClaw", "project")
    e2 = graph.upsert_entity("OpenClaw", "project", aliases=["gateway"])
    assert e1.id == e2.id, "upsert should match by name and return the same id"
    assert "gateway" in e2.aliases


def test_find_entity_by_alias(graph):
    graph.upsert_entity("Duckets", "person", aliases=["Ryan"])
    found = graph.find_entity("Ryan")
    assert found is not None
    assert found.name == "Duckets"


def test_add_and_query_active_relationship(graph):
    duckets = graph.upsert_entity("Duckets", "person")
    openclaw = graph.upsert_entity("OpenClaw", "project")
    rel = graph.add_relationship(duckets.id, openclaw.id, "maintains")
    assert rel.id
    assert rel.is_active
    active = graph.query_active(entity_id=duckets.id)
    assert len(active) == 1
    assert active[0].label == "maintains"


def test_relationship_validity_window(graph):
    a = graph.upsert_entity("Alice", "person")
    b = graph.upsert_entity("Bob", "person")
    # Relationship was true from 30 days ago to 29 days ago (1-day window)
    past = time.time() - 86400 * 30
    rel = graph.add_relationship(a.id, b.id, "worked_with", valid_from=past, valid_until=past + 86400)
    # Now it's 30 days later; relationship should be inactive
    assert not rel.is_active, "expired relationship should not be active"
    # But it was true 29.5 days ago (mid-window)
    mid_window = past + 43200  # 12 hours into the 1-day window
    assert rel.is_active_at(mid_window), "should be active mid-window"
    # Not yet true 31 days ago
    before = time.time() - 86400 * 31
    assert not rel.is_active_at(before), "should not be active before valid_from"
    # Also not true at 28.5 days ago (after valid_until)
    after = past + 86400 + 1
    assert not rel.is_active_at(after), "should not be active after valid_until"


def test_query_at_returns_only_valid_relationships(graph):
    a = graph.upsert_entity("A", "person")
    b = graph.upsert_entity("B", "person")
    c = graph.upsert_entity("C", "person")
    # A→B active
    graph.add_relationship(a.id, b.id, "knows", valid_from=time.time() - 86400)
    # A→C ended 2 days ago (use a clearly past window)
    two_days_ago = time.time() - 86400 * 2
    graph.add_relationship(
        a.id, c.id, "knew",
        valid_from=time.time() - 86400 * 30,
        valid_until=two_days_ago,
    )
    # 3 days ago: A→C was still active, A→B not yet
    three_days_ago = time.time() - 86400 * 3
    historical = graph.query_at(a.id, at=three_days_ago)
    labels = sorted(r.label + "→" + ("B" if r.target_id == b.id else "C") for r in historical)
    assert "knew→C" in labels
    assert "knows→B" not in labels
    # Now: A→B active, A→C not
    current = graph.query_at(a.id, at=time.time())
    labels_now = sorted(r.label + "→" + ("B" if r.target_id == b.id else "C") for r in current)
    assert "knows→B" in labels_now
    assert "knew→C" not in labels_now


def test_history_includes_ended_relationships(graph):
    a = graph.upsert_entity("A", "person")
    b = graph.upsert_entity("B", "person")
    r1 = graph.add_relationship(a.id, b.id, "knew", valid_from=time.time() - 1000, valid_until=time.time() - 500)
    r2 = graph.add_relationship(a.id, b.id, "knows", valid_from=time.time() - 500)
    hist = graph.history(a.id)
    assert len(hist) == 2
    # Newest first
    assert hist[0].valid_from >= hist[1].valid_from


def test_supersede_ends_old_and_creates_new(graph):
    a = graph.upsert_entity("Alice", "person")
    orion = graph.upsert_entity("Orion", "project")
    nebula = graph.upsert_entity("Nebula", "project")
    r_old = graph.add_relationship(a.id, orion.id, "works_on", valid_from=time.time() - 86400)
    new = graph.supersede(r_old.id, a.id, nebula.id, "works_on", at=time.time())
    # Old is ended (refresh from DB)
    r_old_refreshed = graph.get_relationship(r_old.id)
    assert r_old_refreshed is not None
    assert not r_old_refreshed.is_active, "supersede should end the old relationship"
    # New is active
    assert new.is_active
    # Query now should show only the new one
    active = graph.query_active(entity_id=a.id)
    assert len(active) == 1
    assert active[0].target_id == nebula.id


def test_end_relationship(graph):
    a = graph.upsert_entity("A", "person")
    b = graph.upsert_entity("B", "person")
    r = graph.add_relationship(a.id, b.id, "knows")
    assert r.is_active
    ended = graph.end_relationship(r.id)
    assert ended
    # Refresh from DB
    r2 = graph.get_relationship(r.id)
    assert r2 is not None
    assert not r2.is_active


def test_dedupe_active_relationship(graph):
    """If we try to add the same active relationship twice, it should be a no-op."""
    a = graph.upsert_entity("A", "person")
    b = graph.upsert_entity("B", "person")
    r1 = graph.add_relationship(a.id, b.id, "knows")
    r2 = graph.add_relationship(a.id, b.id, "knows")
    assert r1.id == r2.id
    active = graph.query_active(entity_id=a.id)
    assert len(active) == 1


def test_list_entities_by_kind(graph):
    graph.upsert_entity("Alice", "person")
    graph.upsert_entity("Bob", "person")
    graph.upsert_entity("OpenClaw", "project")
    persons = graph.list_entities(kind="person")
    projects = graph.list_entities(kind="project")
    assert len(persons) == 2
    assert len(projects) == 1
    assert projects[0].name == "OpenClaw"


def test_stats(graph):
    a = graph.upsert_entity("A", "person")
    b = graph.upsert_entity("B", "person")
    graph.upsert_entity("Proj", "project")
    graph.add_relationship(a.id, b.id, "knows")
    r2 = graph.add_relationship(a.id, b.id, "knew")
    graph.end_relationship(r2.id)
    s = graph.stats()
    assert s["entities"] == 3
    assert s["relationships"] == 2
    assert s["active_relationships"] == 1
    assert s["ended_relationships"] == 1
    assert s["entities_by_kind"]["person"] == 2
    assert s["entities_by_kind"]["project"] == 1


def test_query_active_filters_by_label(graph):
    a = graph.upsert_entity("A", "person")
    b = graph.upsert_entity("B", "person")
    c = graph.upsert_entity("Proj", "project")
    graph.add_relationship(a.id, b.id, "knows")
    graph.add_relationship(a.id, c.id, "works_on")
    knows = graph.query_active(entity_id=a.id, label="knows")
    assert len(knows) == 1
    assert knows[0].label == "knows"


def test_cascade_delete_removes_relationships(graph):
    """Deleting an entity should cascade to its relationships."""
    a = graph.upsert_entity("A", "person")
    b = graph.upsert_entity("B", "person")
    graph.add_relationship(a.id, b.id, "knows")
    assert len(graph.query_active()) == 1
    graph.delete_entity(a.id)
    assert len(graph.query_active()) == 0


def test_context_manager(tmp_path):
    """Graph can be used as a context manager."""
    db = tmp_path / "ctx-graph.db"
    with Graph(path=db) as g:
        e = g.upsert_entity("X", "thing")
        assert e.id
