"""
graph.py — Temporal knowledge graph for the DuckBot brain.

Inspired by Zep's Graphiti (arXiv:2501.13956) and Cognee's typed graphs.
A knowledge graph here is:

  Node   = an Entity (person, project, file, place, concept, fact)
  Edge   = a Relationship between two entities, with a validity window

The key difference from a plain graph: every edge has
  valid_from  — when the relationship became true
  valid_until — when it stopped being true (None = still true)

This lets us answer "what was true on date X?" and "when did Y change?"
which plain RAG can't.

Storage: SQLite (separate file from chroma, no lock contention).
Embeddings: optional — we store entity text + IDs, and let the existing
chroma store do semantic search. The graph is the structure; chroma is
the meaning.

No LLM is required to use this module. Entity extraction is layered on top
in `entities.py` (Layer 2) and is itself LLM-optional.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional


# ---------------------------------------------------------------------------
# Default storage path: alongside other RAG data
# ---------------------------------------------------------------------------
DEFAULT_GRAPH_PATH = Path(__file__).resolve().parent.parent / "data" / "graph.db"


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """A node in the knowledge graph."""
    id: str
    name: str
    kind: str               # "person" | "project" | "file" | "place" | "concept" | "fact"
    aliases: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    notes: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "aliases": list(self.aliases),
            "created_at": self.created_at,
            "notes": self.notes,
        }


@dataclass
class Relationship:
    """A directed, time-bounded edge between two entities."""
    id: str
    source_id: str
    target_id: str
    label: str              # e.g. "works_on", "created", "located_in", "depends_on"
    valid_from: float       # unix epoch seconds
    valid_until: Optional[float] = None   # None = still true
    confidence: float = 1.0
    source: Optional[str] = None          # which chunk/file the fact came from
    created_at: float = field(default_factory=time.time)

    @property
    def is_active(self) -> bool:
        """True if this relationship is currently valid (no `at` arg)."""
        return self.is_active_at(time.time())

    def is_active_at(self, at: float) -> bool:
        """True if this relationship is valid at the given unix epoch time."""
        if at < self.valid_from:
            return False
        if self.valid_until is not None and at >= self.valid_until:
            return False
        return True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "label": self.label,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "confidence": self.confidence,
            "source": self.source,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS entities (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    kind       TEXT NOT NULL,
    aliases    TEXT,            -- JSON array
    notes      TEXT,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_entities_kind ON entities(kind);

CREATE TABLE IF NOT EXISTS relationships (
    id          TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL,
    target_id   TEXT NOT NULL,
    label       TEXT NOT NULL,
    valid_from  REAL NOT NULL,
    valid_until REAL,
    confidence  REAL NOT NULL DEFAULT 1.0,
    source      TEXT,
    created_at  REAL NOT NULL,
    FOREIGN KEY (source_id) REFERENCES entities(id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES entities(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_id);
CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_id);
CREATE INDEX IF NOT EXISTS idx_rel_label  ON relationships(label);
CREATE INDEX IF NOT EXISTS idx_rel_window ON relationships(valid_from, valid_until);
"""


# ---------------------------------------------------------------------------
# The Graph itself
# ---------------------------------------------------------------------------

class Graph:
    """Temporal knowledge graph backed by SQLite."""

    def __init__(self, path: str | Path = DEFAULT_GRAPH_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # -- Entity ops ---------------------------------------------------------

    def upsert_entity(self, name: str, kind: str, aliases: Iterable[str] = (),
                       notes: Optional[str] = None, entity_id: Optional[str] = None) -> Entity:
        """Insert or update an entity by name. If an entity with this name
        already exists, return the existing one (and merge aliases)."""
        existing = self._find_entity_by_name(name)
        if existing is not None:
            new_aliases = list(set(existing.aliases) | set(aliases))
            self._conn.execute(
                "UPDATE entities SET aliases = ? WHERE id = ?",
                (self._encode_json(new_aliases), existing.id),
            )
            self._conn.commit()
            existing.aliases = new_aliases
            if notes is not None:
                existing.notes = notes
                self._conn.execute(
                    "UPDATE entities SET notes = ? WHERE id = ?",
                    (notes, existing.id),
                )
                self._conn.commit()
            return existing
        eid = entity_id or str(uuid.uuid4())
        ent = Entity(id=eid, name=name, kind=kind, aliases=list(aliases), notes=notes)
        self._conn.execute(
            "INSERT INTO entities (id, name, kind, aliases, notes, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (ent.id, ent.name, ent.kind, self._encode_json(ent.aliases), ent.notes, ent.created_at),
        )
        self._conn.commit()
        return ent

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        row = self._conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        return self._row_to_entity(row) if row else None

    def find_entity(self, name: str) -> Optional[Entity]:
        return self._find_entity_by_name(name)

    def list_entities(self, kind: Optional[str] = None, limit: int = 1000) -> list[Entity]:
        if kind:
            rows = self._conn.execute(
                "SELECT * FROM entities WHERE kind = ? ORDER BY name LIMIT ?",
                (kind, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM entities ORDER BY name LIMIT ?", (limit,)
            ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    def delete_entity(self, entity_id: str) -> bool:
        cur = self._conn.execute("DELETE FROM entities WHERE id = ?", (entity_id,))
        self._conn.commit()
        return cur.rowcount > 0

    # -- Relationship ops ---------------------------------------------------

    def add_relationship(self, source_id: str, target_id: str, label: str,
                          valid_from: Optional[float] = None,
                          valid_until: Optional[float] = None,
                          confidence: float = 1.0,
                          source: Optional[str] = None,
                          relationship_id: Optional[str] = None) -> Relationship:
        """Add a new relationship. If an identical active relationship already
        exists (same source, target, label, no end date), this is a no-op and
        the existing relationship is returned."""
        if valid_from is None:
            valid_from = time.time()
        # Dedupe: if there's already an active edge with the same endpoints+label, reuse it
        existing = self._conn.execute(
            "SELECT * FROM relationships WHERE source_id = ? AND target_id = ? "
            "AND label = ? AND valid_until IS NULL",
            (source_id, target_id, label),
        ).fetchone()
        if existing:
            return self._row_to_relationship(existing)
        rid = relationship_id or str(uuid.uuid4())
        rel = Relationship(
            id=rid,
            source_id=source_id,
            target_id=target_id,
            label=label,
            valid_from=valid_from,
            valid_until=valid_until,
            confidence=confidence,
            source=source,
        )
        self._conn.execute(
            "INSERT INTO relationships (id, source_id, target_id, label, valid_from, "
            "valid_until, confidence, source, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (rel.id, rel.source_id, rel.target_id, rel.label, rel.valid_from,
             rel.valid_until, rel.confidence, rel.source, rel.created_at),
        )
        self._conn.commit()
        return rel

    def end_relationship(self, relationship_id: str, at: Optional[float] = None) -> bool:
        """Mark a relationship as no longer valid (set valid_until)."""
        if at is None:
            at = time.time()
        cur = self._conn.execute(
            "UPDATE relationships SET valid_until = ? WHERE id = ? AND valid_until IS NULL",
            (at, relationship_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def get_relationship(self, relationship_id: str) -> Optional[Relationship]:
        """Fetch a single relationship by id (useful for refreshing an
        in-memory copy after end_relationship / supersede)."""
        row = self._conn.execute(
            "SELECT * FROM relationships WHERE id = ?", (relationship_id,)
        ).fetchone()
        return self._row_to_relationship(row) if row else None

    def supersede(self, old_relationship_id: str, new_source_id: str,
                   new_target_id: str, new_label: str,
                   at: Optional[float] = None) -> Relationship:
        """End an old relationship and start a new one in a single atomic step.
        Useful for 'Kai left project Orion' + 'Kai joined project Nebula'."""
        if at is None:
            at = time.time()
        self.end_relationship(old_relationship_id, at=at)
        return self.add_relationship(
            new_source_id, new_target_id, new_label, valid_from=at, source=None
        )

    def query_active(self, entity_id: Optional[str] = None,
                      at: Optional[float] = None,
                      label: Optional[str] = None) -> list[Relationship]:
        """Return relationships that are valid at time `at` (default: now).
        If entity_id is given, only relationships touching that entity."""
        if at is None:
            at = time.time()
        if entity_id is not None:
            rows = self._conn.execute(
                "SELECT * FROM relationships "
                "WHERE (source_id = ? OR target_id = ?) "
                "AND valid_from <= ? AND (valid_until IS NULL OR valid_until > ?) "
                + ("AND label = ? " if label else "")
                + "ORDER BY valid_from DESC",
                (entity_id, entity_id, at, at) + ((label,) if label else ()),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM relationships "
                "WHERE valid_from <= ? AND (valid_until IS NULL OR valid_until > ?) "
                + ("AND label = ? " if label else "")
                + "ORDER BY valid_from DESC",
                (at, at) + ((label,) if label else ()),
            ).fetchall()
        return [self._row_to_relationship(r) for r in rows]

    def query_at(self, entity_id: str, at: float) -> list[Relationship]:
        """What was true about `entity_id` at time `at`?"""
        return self.query_active(entity_id=entity_id, at=at)

    def history(self, entity_id: str) -> list[Relationship]:
        """All relationships (active + ended) touching this entity, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM relationships WHERE source_id = ? OR target_id = ? "
            "ORDER BY valid_from DESC",
            (entity_id, entity_id),
        ).fetchall()
        return [self._row_to_relationship(r) for r in rows]

    # -- Stats / maintenance -----------------------------------------------

    def stats(self) -> dict:
        n_ent = self._conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        n_rel = self._conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]
        n_active = self._conn.execute(
            "SELECT COUNT(*) FROM relationships WHERE valid_until IS NULL"
        ).fetchone()[0]
        by_kind = dict(self._conn.execute(
            "SELECT kind, COUNT(*) FROM entities GROUP BY kind"
        ).fetchall())
        return {
            "entities": n_ent,
            "relationships": n_rel,
            "active_relationships": n_active,
            "ended_relationships": n_rel - n_active,
            "entities_by_kind": by_kind,
        }

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Graph":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- Internals ----------------------------------------------------------

    def _find_entity_by_name(self, name: str) -> Optional[Entity]:
        # exact name match first
        row = self._conn.execute(
            "SELECT * FROM entities WHERE name = ?", (name,)
        ).fetchone()
        if row:
            return self._row_to_entity(row)
        # then alias match
        rows = self._conn.execute("SELECT * FROM entities WHERE aliases IS NOT NULL").fetchall()
        for r in rows:
            aliases = self._decode_json(r["aliases"]) or []
            if name in aliases:
                return self._row_to_entity(r)
        return None

    def _row_to_entity(self, row: sqlite3.Row) -> Entity:
        return Entity(
            id=row["id"],
            name=row["name"],
            kind=row["kind"],
            aliases=self._decode_json(row["aliases"]) or [],
            notes=row["notes"],
            created_at=row["created_at"],
        )

    def _row_to_relationship(self, row: sqlite3.Row) -> Relationship:
        return Relationship(
            id=row["id"],
            source_id=row["source_id"],
            target_id=row["target_id"],
            label=row["label"],
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            confidence=row["confidence"],
            source=row["source"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _encode_json(obj) -> str:
        import json
        return json.dumps(obj, ensure_ascii=False)

    @staticmethod
    def _decode_json(s) -> Optional[object]:
        import json
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None
