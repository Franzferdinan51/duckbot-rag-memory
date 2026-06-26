"""observer.py — causal precursor tracing + blind-spot detection.

Inspired by MindBank's "Observer Perspective: Causal Intelligence"
which walks the entity graph backward to surface *why* we know
something, and flags orphan decisions whose reasoning chain is
missing.

Re-implemented natively against duckbot's `src/graph.py`. Two public
functions:

    trace_precursors(graph, entity_name, ...) -> dict
        Backward BFS from `entity_name` through causal edge labels
        (decided_by, depends_on, learned_from, caused_by, related_to).
        Returns a depth-labeled tree of precursors + summary stats:
        - critical_depth:    shallowest depth capturing >=90% of influence
        - coverage:          fraction of immediate edges with upstream
        - influence_modes:   top precursors ranked by score
        - chain:             depth-indexed list of precursors

    find_blind_spots(graph, ...) -> list[dict]
        Identifies entities with causal edges but no upstream
        precursors. These are decisions / facts whose reasoning
        chain is missing — the agent thinks "we use Postgres" but
        can't trace *why* anymore. Operators should either backfill
        with decided_by / learned_from edges, or mark the decision
        as deliberate (no precursor needed).

Pure functions; no I/O. The Brain facade wires them in
(src/connectors/base.py:graph_precursors / graph_blind_spots).
Licensed MIT (this file is original work; design borrowed from
MindBank, also MIT).
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


# Edge labels that represent causal dependency (vs purely relational).
# When tracing "why do we know X?", we walk backward through these.
CAUSAL_LABELS: frozenset[str] = frozenset({
    "decided_by",
    "depends_on",
    "learned_from",
    "caused_by",
    "related_to",       # soft causal: "this is related to that"
    "supports",         # one fact supports another
    "contradicts",      # important for blind-spot detection
})

# How fast does influence decay with depth? Higher = faster decay.
# At depth 1: 1.0. At depth 2: 0.5. At depth 3: 0.25. (half-life ~1 hop)
INFLUENCE_DECAY_PER_DEPTH = 0.5


@dataclass
class PrecursorNode:
    """A precursor found during backward BFS."""
    entity_id: str           # graph stores entity IDs as strings (slugified names)
    entity_name: str
    depth: int
    via_label: str          # which edge label led us here
    via_relationship_id: str
    influence_score: float  # decayed by depth
    valid_from: Optional[float] = None
    valid_until: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "depth": self.depth,
            "via_label": self.via_label,
            "via_relationship_id": self.via_relationship_id,
            "influence_score": round(self.influence_score, 4),
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
        }


@dataclass
class PrecursorTrace:
    """Result of trace_precursors()."""
    root: str
    root_entity_id: str            # graph stores IDs as strings
    total_nodes: int
    max_depth_reached: int
    critical_depth: int            # shallowest depth >= 90% of influence
    coverage: float                # fraction of immediate edges w/ upstream
    immediate_edge_count: int
    precursors_with_upstream: int
    chain: list[list[PrecursorNode]] = field(default_factory=list)
    influence_modes: list[PrecursorNode] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "root_entity_id": self.root_entity_id,
            "total_nodes": self.total_nodes,
            "max_depth_reached": self.max_depth_reached,
            "critical_depth": self.critical_depth,
            "coverage": round(self.coverage, 4),
            "immediate_edge_count": self.immediate_edge_count,
            "precursors_with_upstream": self.precursors_with_upstream,
            "chain": [
                [p.to_dict() for p in layer] for layer in self.chain
            ],
            "influence_modes": [p.to_dict() for p in self.influence_modes],
            "notes": self.notes,
        }


def trace_precursors(
    graph,
    entity_name: str,
    *,
    max_depth: int = 3,
    causal_labels: Iterable[str] = CAUSAL_LABELS,
    include_inactive: bool = False,
    min_influence: float = 0.0,
) -> PrecursorTrace:
    """Backward BFS from `entity_name` through causal edges.

    Args:
      graph: a `src.graph.Graph` instance (open inside a `with` block
        or passed directly — we only read).
      entity_name: the entity to trace from (case-insensitive).
      max_depth: BFS depth limit (default 3 — sufficient for most
        personal-agent corpora without exploding the result).
      causal_labels: which edge labels count as causal. Default
        `CAUSAL_LABELS` (decided_by / depends_on / learned_from /
        caused_by / related_to / supports / contradicts).
      include_inactive: include relationships whose `valid_until`
        has passed. Default False — only "active" relationships
        contribute to the causal chain.
      min_influence: drop precursors whose decayed influence score
        falls below this floor (default 0.0 = keep everything).

    Returns:
      PrecursorTrace with the depth-indexed chain + summary stats.
    """
    causal_set = frozenset(l.lower() for l in causal_labels)
    notes: list[str] = []

    root_entity = graph.find_entity(entity_name)
    if root_entity is None:
        notes.append(f"entity not found: {entity_name!r}")
        return PrecursorTrace(
            root=entity_name,
            root_entity_id="",
            total_nodes=0,
            max_depth_reached=0,
            critical_depth=0,
            coverage=0.0,
            immediate_edge_count=0,
            precursors_with_upstream=0,
            notes=notes,
        )

    # 1. Find immediate causal predecessors (depth 1). Use
    # `graph.history()` (active + ended) when include_inactive is True,
    # `graph.query_active()` (just active) otherwise. The Graph API
    # has no single `relationships_involving(id)` method — `history`
    # and `query_active` both filter to "this entity is source OR
    # target" when called with an entity_id.
    if include_inactive:
        all_rels = graph.history(root_entity.id)
    else:
        all_rels = graph.query_active(entity_id=root_entity.id)

    immediate: list = [
        r for r in all_rels
        if r.target_id == root_entity.id
        and (r.label or "").lower() in causal_set
    ]
    immediate_count = len(immediate)
    if immediate_count == 0:
        notes.append(
            "no causal edges pointing into this entity — nothing to trace"
        )
        return PrecursorTrace(
            root=root_entity.name,
            root_entity_id=root_entity.id,
            total_nodes=0,
            max_depth_reached=0,
            critical_depth=0,
            coverage=0.0,
            immediate_edge_count=0,
            precursors_with_upstream=0,
            notes=notes,
        )

    # 2. BFS backward, depth-first by relationship_id (deterministic
    # order so tests don't flake).
    chain: list[list[PrecursorNode]] = [[] for _ in range(max_depth)]
    seen: set[int] = set()
    seen.add(root_entity.id)
    queue: deque = deque()

    # Seed the queue with depth-1 predecessors.
    for r in immediate:
        if r.source_id in seen:
            continue
        seen.add(r.source_id)
        queue.append((r.source_id, 1, r.id, r.label, r.valid_from, r.valid_until))

    total_influence: dict[int, float] = defaultdict(float)

    while queue:
        cur_id, depth, via_rel_id, via_label, vf, vu = queue.popleft()
        if depth > max_depth:
            break
        influence = INFLUENCE_DECAY_PER_DEPTH ** (depth - 1)
        if influence < min_influence:
            continue
        # Look up the entity (may have been deleted — handle gracefully).
        ent = graph.get_entity(cur_id) if hasattr(graph, "get_entity") else None
        name = ent.name if ent is not None else f"<id={cur_id}>"
        node = PrecursorNode(
            entity_id=cur_id,
            entity_name=name,
            depth=depth,
            via_label=via_label or "",
            via_relationship_id=via_rel_id,
            influence_score=influence,
            valid_from=vf,
            valid_until=vu,
        )
        chain[depth - 1].append(node)
        total_influence[depth] += influence

        # Continue BFS: find this node's own predecessors.
        if depth < max_depth:
            if include_inactive:
                sub_rels = graph.history(cur_id)
            else:
                sub_rels = graph.query_active(entity_id=cur_id)
            for r in sub_rels:
                if r.target_id != cur_id:
                    continue
                if (r.label or "").lower() not in causal_set:
                    continue
                if r.source_id in seen:
                    continue
                seen.add(r.source_id)
                queue.append(
                    (r.source_id, depth + 1, r.id, r.label, r.valid_from, r.valid_until)
                )

    # 3. Compute summary stats.
    flat = [n for layer in chain for n in layer]
    max_depth_reached = max((n.depth for n in flat), default=0)
    influence_modes = sorted(flat, key=lambda n: n.influence_score, reverse=True)

    # Critical depth: shallowest depth whose cumulative influence is
    # >= 90% of the total accumulated influence.
    total_inf = sum(n.influence_score for n in flat) or 1.0
    cumulative = 0.0
    critical_depth = max_depth_reached
    for depth in range(1, max_depth_reached + 1):
        layer_inf = sum(n.influence_score for n in chain[depth - 1])
        cumulative += layer_inf
        if cumulative / total_inf >= 0.9:
            critical_depth = depth
            break
    else:
        # If we never hit 90%, report the max depth reached (the chain
        # is shallow enough that everything is already critical).
        critical_depth = max_depth_reached

    # Coverage: fraction of immediate causal edges that have an
    # upstream precursor we could trace. We approximate "upstream exists"
    # by whether the source entity of the edge appears anywhere in
    # the chain — entity was created by some other act, so we look
    # for a "second hop" into it.
    coverage = 0.0
    if immediate_count > 0:
        precursors_with_upstream = 0
        for r in immediate:
            # Find the depth-1 node for this edge, then check whether
            # that node has any depth-2 precursor.
            d1_match = next(
                (n for n in chain[0] if n.via_relationship_id == r.id),
                None,
            )
            if d1_match is not None and len(chain) > 1 and chain[1]:
                precursors_with_upstream += 1
        coverage = precursors_with_upstream / immediate_count
    else:
        precursors_with_upstream = 0

    if max_depth_reached == 1 and immediate_count > 0:
        notes.append(
            "depth 1 only — increase max_depth to see the full reasoning chain"
        )
    if coverage < 0.5 and immediate_count > 0:
        notes.append(
            f"low coverage ({coverage:.0%}) — "
            f"{immediate_count - precursors_with_upstream} of "
            f"{immediate_count} causal edges lack upstream rationale"
        )

    return PrecursorTrace(
        root=root_entity.name,
        root_entity_id=root_entity.id,
        total_nodes=len(flat),
        max_depth_reached=max_depth_reached,
        critical_depth=critical_depth,
        coverage=coverage,
        immediate_edge_count=immediate_count,
        precursors_with_upstream=precursors_with_upstream,
        chain=chain,
        influence_modes=influence_modes,
        notes=notes,
    )


@dataclass
class BlindSpot:
    """An entity that has causal edges but no upstream rationale."""
    entity_id: str
    entity_name: str
    causal_edge_count: int
    severity: str             # "low" | "medium" | "high"
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "entity_name": self.entity_name,
            "causal_edge_count": self.causal_edge_count,
            "severity": self.severity,
            "reason": self.reason,
        }


def find_blind_spots(
    graph,
    *,
    causal_labels: Iterable[str] = CAUSAL_LABELS,
    min_outgoing_causal: int = 1,
    max_results: int = 50,
    include_inactive: bool = False,
) -> list[BlindSpot]:
    """Find entities that have causal edges but no upstream precursor.

    These are the "orphan decisions" — facts whose reasoning chain
    is missing. The agent may believe them, but can't answer *why*.

    Severity:
      high   — entity has >= 3 outgoing causal edges with no upstream
      medium — entity has 2 outgoing causal edges with no upstream
      low    — entity has 1 outgoing causal edge with no upstream

    Args:
      graph: a `src.graph.Graph` instance.
      causal_labels: which edge labels count as causal.
      min_outgoing_causal: ignore entities with fewer than this many
        outgoing causal edges. Default 1 = any entity with a decided_by /
        depends_on / learned_from edge is a candidate.
      max_results: cap on returned blind spots (sorted by severity).
      include_inactive: include entities whose relationships are
        inactive. Default False.

    Returns:
      list of BlindSpot, sorted by severity (high → low) then
      causal_edge_count desc.
    """
    causal_set = frozenset(l.lower() for l in causal_labels)
    spots: list[BlindSpot] = []

    for ent in graph.list_entities():
        if include_inactive:
            rels = graph.history(ent.id)
        else:
            rels = graph.query_active(entity_id=ent.id)
        # Outgoing causal edges: this entity is the SOURCE of a
        # decided_by / depends_on / learned_from edge.
        outgoing_causal = [
            r for r in rels
            if r.source_id == ent.id
            and (r.label or "").lower() in causal_set
        ]
        if len(outgoing_causal) < min_outgoing_causal:
            continue
        # Incoming causal edges: precursors. If any exist, this entity
        # IS someone's upstream rationale and the chain is intact.
        incoming_causal = [
            r for r in rels
            if r.target_id == ent.id
            and (r.label or "").lower() in causal_set
        ]
        if incoming_causal:
            continue
        n = len(outgoing_causal)
        if n >= 3:
            severity = "high"
            reason = (
                f"{n} downstream decisions depend on this entity, but "
                f"no upstream rationale (no decided_by / depends_on / "
                f"learned_from edge points INTO it)"
            )
        elif n == 2:
            severity = "medium"
            reason = (
                f"2 downstream edges with no upstream rationale"
            )
        else:
            severity = "low"
            reason = (
                f"1 downstream edge with no upstream rationale"
            )
        spots.append(BlindSpot(
            entity_id=ent.id,
            entity_name=ent.name,
            causal_edge_count=n,
            severity=severity,
            reason=reason,
        ))

    severity_rank = {"high": 0, "medium": 1, "low": 2}
    spots.sort(
        key=lambda s: (severity_rank.get(s.severity, 9), -s.causal_edge_count)
    )
    return spots[:max_results]


__all__ = [
    "CAUSAL_LABELS",
    "INFLUENCE_DECAY_PER_DEPTH",
    "PrecursorNode",
    "PrecursorTrace",
    "BlindSpot",
    "trace_precursors",
    "find_blind_spots",
]