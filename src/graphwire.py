"""
graphwire.py — wiring for the temporal knowledge graph (Phase C).

src/graph.py + src/entities.py have implemented a Graphiti-style
temporal graph (entities, relationships with validity windows, SQLite at
data/graph.db) since long before this upgrade — but nothing fed it and
recall never consulted it; only the dashboard imported it. This module
is the missing wire:

  feed_text(text)        — extract entities/triples (regex ECL, no LLM)
                           and cognify them into the graph. Idempotent.
  neighbors_for_query(q) — active relationship lines for entities named
                           in the query ("Duckets — lives_in → Dayton").

Path: DUCKBOT_GRAPH_PATH env or graph.DEFAULT_GRAPH_PATH. Every public
function is fail-open — a graph problem must never break a memory write
or a recall, so callers use the *_safe variants or catch.
Import as src.graphwire.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_GRAPH = None
_GRAPH_PATH: Optional[str] = None


def graph_path() -> Path:
    env = os.environ.get("DUCKBOT_GRAPH_PATH")
    if env:
        return Path(env)
    from .graph import DEFAULT_GRAPH_PATH
    return Path(DEFAULT_GRAPH_PATH)


def get_graph():
    """Lazily build (and cache) the Graph for the configured path."""
    global _GRAPH, _GRAPH_PATH
    p = str(graph_path())
    if _GRAPH is None or _GRAPH_PATH != p:
        from .graph import Graph
        _GRAPH = Graph(p)
        _GRAPH_PATH = p
    return _GRAPH


def feed_text(text: str, source: Optional[str] = None) -> dict:
    """Extract entities/triples from text and load them into the graph."""
    if not text or not text.strip():
        return {}
    from .entities import EntityExtractor
    extractor = EntityExtractor()
    entities, triples = extractor.extract(text, source=source)
    if not entities and not triples:
        return {}
    return extractor.cognify(get_graph(), entities, triples, source=source)


def feed_text_safe(text: str, source: Optional[str] = None) -> dict:
    try:
        return feed_text(text, source=source)
    except Exception as exc:  # fail-open: graph never breaks a write
        logger.warning("graph feed failed (%s)", exc)
        return {}


def neighbors_for_query(query: str, *, limit: int = 10) -> list[str]:
    """Active relationship lines for entities mentioned in the query."""
    if not query or not query.strip():
        return []
    g = get_graph()
    q = query.lower()
    lines: list[str] = []
    seen: set[str] = set()
    for ent in g.list_entities(limit=2000):
        name = (ent.name or "").strip()
        if len(name) < 3 or name.lower() not in q:
            continue
        for rel in g.query_active(entity_id=ent.id):
            other_id = rel.target_id if rel.source_id == ent.id else rel.source_id
            other = g.get_entity(other_id)
            other_name = other.name if other else other_id
            line = f"{name} — {rel.label} → {other_name}"
            if line not in seen:
                seen.add(line)
                lines.append(line)
            if len(lines) >= limit:
                return lines
    return lines
