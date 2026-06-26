"""Tests for src/observer.py — causal precursor tracing + blind-spot detection."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.graph import Graph  # noqa: E402
from src.observer import (  # noqa: E402
    CAUSAL_LABELS,
    INFLUENCE_DECAY_PER_DEPTH,
    PrecursorTrace,
    trace_precursors,
    find_blind_spots,
)


@pytest.fixture
def graph(tmp_path):
    """A fresh graph.db for each test."""
    return Graph(path=tmp_path / "graph.db")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_chain(graph: Graph) -> dict[str, str]:
    """Build a 3-hop causal chain. Returns the entity id map.

    Layout (source → target, all causal labels):

        load_test_results ───decided_by──→ use_postgres
        use_postgres       ───depends_on──→ acid_requirements
        acid_requirements  ───supports───→ financial_integrity

    So tracing BACKWARD from `financial_integrity` should reach all
    three upstream nodes (1 hop each).
    """
    load = graph.upsert_entity("Load Test Results", kind="fact")
    pg = graph.upsert_entity("Use Postgres", kind="decision")
    acid = graph.upsert_entity("ACID Requirements", kind="requirement")
    fin = graph.upsert_entity("Financial Data Integrity", kind="requirement")

    graph.add_relationship(load.id, pg.id, "decided_by")
    graph.add_relationship(pg.id, acid.id, "depends_on")
    graph.add_relationship(acid.id, fin.id, "supports")

    return {
        "load": load.id,
        "pg": pg.id,
        "acid": acid.id,
        "fin": fin.id,
    }


# ---------------------------------------------------------------------------
# Constants + module surface
# ---------------------------------------------------------------------------


def test_causal_labels_default_set():
    """The default causal labels must include the canonical four."""
    expected = {"decided_by", "depends_on", "learned_from", "caused_by"}
    assert expected <= CAUSAL_LABELS


def test_influence_decay_constant():
    assert 0.0 < INFLUENCE_DECAY_PER_DEPTH < 1.0
    assert INFLUENCE_DECAY_PER_DEPTH == 0.5


# ---------------------------------------------------------------------------
# trace_precursors — happy path
# ---------------------------------------------------------------------------


def test_trace_precursors_unknown_entity_returns_empty(graph):
    """Entity not in the graph → empty trace with a note."""
    trace = trace_precursors(graph, "nonexistent_entity")
    assert trace.root == "nonexistent_entity"
    assert trace.total_nodes == 0
    assert trace.immediate_edge_count == 0
    assert "entity not found" in trace.notes[0]


def test_trace_precursors_no_causal_edges(graph):
    """Entity exists but only has relational (non-causal) edges."""
    graph.upsert_entity("Alice", kind="thing")
    graph.upsert_entity("Bob", kind="thing")
    graph.add_relationship(
        graph.find_entity("Alice").id,
        graph.find_entity("Bob").id,
        "knows",  # relational, not causal
    )
    trace = trace_precursors(graph, "Bob")
    assert trace.total_nodes == 0
    assert "no causal edges" in trace.notes[0]


def test_trace_precursors_one_hop(graph):
    """Depth 1 only — single upstream entity."""
    ids = _build_chain(graph)
    trace = trace_precursors(graph, "Use Postgres")
    assert trace.root == "Use Postgres"
    assert trace.immediate_edge_count == 1
    assert trace.max_depth_reached == 1
    # The one depth-1 node should be "Load Test Results".
    assert len(trace.chain[0]) == 1
    assert trace.chain[0][0].entity_name == "Load Test Results"
    assert trace.chain[0][0].depth == 1
    assert trace.chain[0][0].via_label == "decided_by"
    assert trace.chain[0][0].influence_score == 1.0


def test_trace_precursors_three_hops(graph):
    """Full chain — all three upstream entities reachable."""
    _build_chain(graph)
    # Trace backward from the leaf (Financial Data Integrity). Its
    # chain is ACID → PG → Load, so all three nodes appear at depths
    # 1, 2, 3 respectively.
    trace = trace_precursors(graph, "Financial Data Integrity", max_depth=5)
    assert trace.max_depth_reached == 3
    # All three precursors must appear in the chain (depths 1-3).
    all_names = [n.entity_name for layer in trace.chain for n in layer]
    assert "ACID Requirements" in all_names
    assert "Use Postgres" in all_names
    assert "Load Test Results" in all_names


def test_trace_precursors_respects_max_depth(graph):
    """Setting max_depth=2 stops the BFS at 2 hops."""
    _build_chain(graph)
    # Tracing from fin (leaf): depth 1=acid, depth 2=pg, depth 3=load.
    # With max_depth=2 we get acid and pg but not load.
    trace = trace_precursors(graph, "Financial Data Integrity", max_depth=2)
    assert trace.max_depth_reached == 2
    all_names = [n.entity_name for layer in trace.chain for n in layer]
    assert "ACID Requirements" in all_names
    assert "Use Postgres" in all_names
    assert "Load Test Results" not in all_names


def test_trace_precursors_influence_decays_with_depth(graph):
    """Each depth halves the influence score."""
    _build_chain(graph)
    # Trace from fin (leaf). Chain: acid(1) → pg(2) → load(3).
    trace = trace_precursors(graph, "Financial Data Integrity", max_depth=5)
    by_depth = {}
    for layer in trace.chain:
        for n in layer:
            by_depth.setdefault(n.depth, []).append(n.influence_score)
    assert by_depth[1] == [pytest.approx(1.0)]
    assert by_depth[2] == [pytest.approx(0.5)]
    assert by_depth[3] == [pytest.approx(0.25)]


def test_trace_precursors_critical_depth_first_layer(graph):
    """For a 1-hop trace (only depth-1 has nodes), critical depth = 1.

    The math: total influence = 1.0 (just depth-1). Depth 1 captures
    1.0/1.0 = 100% >= 90%, so critical_depth=1.
    """
    _build_chain(graph)
    # Trace from "Use Postgres" — only Load is upstream (depth 1).
    trace = trace_precursors(graph, "Use Postgres", max_depth=5)
    assert trace.max_depth_reached == 1
    assert trace.critical_depth == 1


def test_trace_precursors_critical_depth_three_hops(graph):
    """For a 3-hop trace, critical depth is the depth that crosses 90%.

    Total influence: 1.0 (d1) + 0.5 (d2) + 0.25 (d3) = 1.75. Cumulative:
    - after d1: 1.0/1.75 = 57%
    - after d2: 1.5/1.75 = 86%
    - after d3: 1.75/1.75 = 100%
    Critical depth = 3 (only after d3 do we cross 90%).
    """
    _build_chain(graph)
    trace = trace_precursors(graph, "Financial Data Integrity", max_depth=5)
    assert trace.critical_depth == 3


def test_trace_precursors_handles_cycles_without_looping_forever(graph):
    """BFS must respect the `seen` set so cycles don't blow up."""
    ids = _build_chain(graph)
    # Add a back-edge: Load Test Results supports Financial Data
    # Integrity (which is the leaf, also reachable from the chain).
    graph.add_relationship(ids["load"], ids["fin"], "supports")
    trace = trace_precursors(graph, "Financial Data Integrity", max_depth=5)
    # Should NOT infinite-loop and root should appear exactly once
    # (at depth 0 = root, never as a precursor).
    all_names = [n.entity_name for layer in trace.chain for n in layer]
    assert all_names.count("Financial Data Integrity") == 0


# ---------------------------------------------------------------------------
# trace_precursors — coverage
# ---------------------------------------------------------------------------


def test_trace_precursors_full_coverage(graph):
    """If depth-2 has nodes, coverage should be 1.0."""
    _build_chain(graph)
    # Tracing from fin (leaf): depth-1 acid has depth-2 upstream (pg).
    trace = trace_precursors(graph, "Financial Data Integrity", max_depth=3)
    assert trace.coverage == pytest.approx(1.0, abs=1e-9)


def test_trace_precursors_zero_coverage_for_orphan_decision(graph):
    """A decision with only ONE hop and nothing upstream has 0% coverage."""
    leaf = graph.upsert_entity("Use MongoDB", kind="decision")
    graph.add_relationship(
        graph.upsert_entity("Fast", kind="thing").id, leaf.id, "decided_by",
    )
    trace = trace_precursors(graph, "Use MongoDB", max_depth=2)
    # Only "Fast" is upstream. Does "Fast" itself have upstream?
    # No → coverage = 0/1 = 0.0.
    assert trace.coverage == 0.0
    assert any("low coverage" in n for n in trace.notes)


def test_trace_precursors_inactive_edges_excluded_by_default(graph):
    """`include_inactive=False` (default) drops ended relationships."""
    ids = _build_chain(graph)
    # End the acid→fin edge (the deepest causal link).
    rels = graph.history(ids["fin"])
    acid_to_fin = next(
        r for r in rels
        if r.source_id == ids["acid"] and r.target_id == ids["fin"]
    )
    graph.end_relationship(acid_to_fin.id)
    # Default: exclude inactive → no depth-1 precursor for fin.
    trace = trace_precursors(graph, "Financial Data Integrity", max_depth=3)
    assert trace.immediate_edge_count == 0


def test_trace_precursors_inactive_edges_included_when_requested(graph):
    """`include_inactive=True` keeps ended relationships in the trace."""
    ids = _build_chain(graph)
    rels = graph.history(ids["fin"])
    acid_to_fin = next(
        r for r in rels
        if r.source_id == ids["acid"] and r.target_id == ids["fin"]
    )
    graph.end_relationship(acid_to_fin.id)
    trace = trace_precursors(
        graph, "Financial Data Integrity", max_depth=3, include_inactive=True,
    )
    assert trace.immediate_edge_count == 1


# ---------------------------------------------------------------------------
# trace_precursors — custom causal labels
# ---------------------------------------------------------------------------


def test_trace_precursors_custom_labels(graph):
    """Custom label set ignores the default decided_by/depends_on."""
    graph.upsert_entity("X", kind="thing")
    y = graph.upsert_entity("Y", kind="thing")
    graph.upsert_entity("Z", kind="thing")
    graph.add_relationship(
        graph.find_entity("X").id, y.id, "decided_by",
    )
    # Z→Y with a non-default label.
    graph.add_relationship(graph.find_entity("Z").id, y.id, "inspired_by")
    # Default trace ignores "inspired_by".
    trace_default = trace_precursors(graph, "Y", max_depth=2)
    assert trace_default.immediate_edge_count == 1  # only decided_by
    # Custom trace picks up both.
    trace_custom = trace_precursors(
        graph, "Y", causal_labels={"decided_by", "inspired_by"}, max_depth=2,
    )
    assert trace_custom.immediate_edge_count == 2


# ---------------------------------------------------------------------------
# find_blind_spots
# ---------------------------------------------------------------------------


def test_find_blind_spots_empty_graph(graph):
    """No entities → no blind spots."""
    assert find_blind_spots(graph) == []


def test_find_blind_spots_no_causal_edges(graph):
    """Entities with only relational edges aren't blind spots."""
    alice = graph.upsert_entity("Alice", kind="thing")
    bob = graph.upsert_entity("Bob", kind="thing")
    graph.add_relationship(alice.id, bob.id, "knows")
    assert find_blind_spots(graph) == []


def test_find_blind_spots_low_severity_single_edge(graph):
    """One downstream edge with no upstream = low severity."""
    fast = graph.upsert_entity("Fast", kind="thing")
    pg = graph.upsert_entity("Use Postgres", kind="thing")
    graph.add_relationship(fast.id, pg.id, "decided_by")
    spots = find_blind_spots(graph)
    assert len(spots) == 1
    assert spots[0].entity_name == "Fast"
    assert spots[0].severity == "low"
    assert spots[0].causal_edge_count == 1


def test_find_blind_spots_medium_severity_two_edges(graph):
    fast = graph.upsert_entity("Fast", kind="thing")
    pg = graph.upsert_entity("Use Postgres", kind="thing")
    rel = graph.upsert_entity("Reliable", kind="thing")
    graph.add_relationship(fast.id, pg.id, "decided_by")
    graph.add_relationship(fast.id, rel.id, "decided_by")
    spots = find_blind_spots(graph)
    assert len(spots) == 1
    assert spots[0].severity == "medium"
    assert spots[0].causal_edge_count == 2


def test_find_blind_spots_high_severity_three_edges(graph):
    fast = graph.upsert_entity("Fast", kind="thing")
    a = graph.upsert_entity("Postgres", kind="thing")
    b = graph.upsert_entity("Mongo", kind="thing")
    c = graph.upsert_entity("Redis", kind="thing")
    for target in (a, b, c):
        graph.add_relationship(fast.id, target.id, "decided_by")
    spots = find_blind_spots(graph)
    assert len(spots) == 1
    assert spots[0].severity == "high"
    assert spots[0].causal_edge_count == 3


def test_find_blind_spots_skips_entities_with_upstream(graph):
    """An entity that HAS incoming causal edges (i.e., itself is
    justified by something upstream) is NOT a blind spot — even if
    it also has outgoing edges.

    Build: a → b (decided_by) → c (decided_by)
    Then b has both incoming (from a) AND outgoing (to c). b is NOT
    a blind spot — its rationale chain goes back to a.
    """
    a = graph.upsert_entity("A", kind="thing")
    b = graph.upsert_entity("B", kind="thing")
    c = graph.upsert_entity("C", kind="thing")
    graph.add_relationship(a.id, b.id, "decided_by")
    graph.add_relationship(b.id, c.id, "decided_by")
    spots = find_blind_spots(graph)
    # a has no outgoing → not a candidate.
    # b has incoming → not a blind spot (correctly excluded).
    # c has no outgoing → not a candidate.
    assert all(s.entity_name != "B" for s in spots)


def test_find_blind_spots_flags_entities_with_only_outgoing(graph):
    """The classic blind spot: entity is a SOURCE of decisions but
    has no upstream rationale of its own."""
    fast = graph.upsert_entity("Fast", kind="thing")
    pg = graph.upsert_entity("Use Postgres", kind="decision")
    graph.add_relationship(fast.id, pg.id, "decided_by")
    spots = find_blind_spots(graph)
    assert len(spots) == 1
    assert spots[0].entity_name == "Fast"
    # Fast has 1 outgoing edge → low severity.
    assert spots[0].severity == "low"


def test_find_blind_spots_sorted_by_severity(graph):
    """High-severity blind spots come first."""
    # Three orphans: severity low/medium/high
    a = graph.upsert_entity("LowRisk", kind="thing")
    b = graph.upsert_entity("MediumRisk", kind="thing")
    c = graph.upsert_entity("HighRisk", kind="thing")
    # 1 edge → low
    graph.add_relationship(a.id, graph.upsert_entity("T1", kind="thing").id, "decided_by")
    # 2 edges → medium
    graph.add_relationship(b.id, graph.upsert_entity("T2", kind="thing").id, "decided_by")
    graph.add_relationship(b.id, graph.upsert_entity("T3", kind="thing").id, "decided_by")
    # 3 edges → high
    graph.add_relationship(c.id, graph.upsert_entity("T4", kind="thing").id, "decided_by")
    graph.add_relationship(c.id, graph.upsert_entity("T5", kind="thing").id, "decided_by")
    graph.add_relationship(c.id, graph.upsert_entity("T6", kind="thing").id, "decided_by")

    spots = find_blind_spots(graph)
    severities = [s.severity for s in spots]
    assert severities == ["high", "medium", "low"]


def test_find_blind_spots_respects_max_results(graph):
    """The cap returns at most N spots."""
    for i in range(10):
        node = graph.upsert_entity(f"orphan-{i}", kind="thing")
        graph.add_relationship(
            node.id, graph.upsert_entity(f"target-{i}", kind="thing").id, "decided_by",
        )
    spots = find_blind_spots(graph, max_results=3)
    assert len(spots) == 3


# ---------------------------------------------------------------------------
# Round-trip — to_dict shape
# ---------------------------------------------------------------------------


def test_trace_to_dict_has_expected_keys(graph):
    _build_chain(graph)
    trace = trace_precursors(graph, "Financial Data Integrity", max_depth=3)
    d = trace.to_dict()
    assert d["root"] == "Financial Data Integrity"
    assert d["total_nodes"] >= 1
    assert isinstance(d["chain"], list)
    assert isinstance(d["influence_modes"], list)
    assert 0.0 <= d["coverage"] <= 1.0
    assert d["max_depth_reached"] >= 1


def test_blind_spot_to_dict_has_expected_keys(graph):
    fast = graph.upsert_entity("Fast", kind="thing")
    pg = graph.upsert_entity("Use Postgres", kind="thing")
    graph.add_relationship(fast.id, pg.id, "decided_by")
    spots = find_blind_spots(graph)
    assert len(spots) == 1
    d = spots[0].to_dict()
    for k in ("entity_id", "entity_name", "causal_edge_count", "severity", "reason"):
        assert k in d
    assert d["severity"] in ("low", "medium", "high")