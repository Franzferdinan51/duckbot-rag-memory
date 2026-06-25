"""
palace.py — Wing/Room/Drawer 2D hierarchy over the brain.

MemPalace structures memory as a 3-level hierarchy:
  WING  (person or project)        — broad category
    ROOM  (day or session)          — time-based grouping
      DRAWER  (verbatim chunk)      — actual content

Our existing tier system (working/episodic/semantic/procedural) is
orthogonal to this — it's a memory-type axis, not a people-project
axis. This module adds the Wing/Room/Drawer view ON TOP of the tier
view so an agent can do "show me everything about OpenClaw from
Tuesday" without filtering manually.

Mapping:
  WING  = derived from a chunk's source_path (e.g. /notes/openclaw.md
          → wing "openclaw") OR the entity graph (entity "OpenClaw").
  ROOM  = the chunk's ingested_at date (ISO YYYY-MM-DD), or the file
          name for static sources.
  DRAWER = the chunk itself (chunk_id + verbatim_text).

Usage:
    from src.palace import PalaceIndex
    pi = PalaceIndex.from_store(store)
    drawers = pi.walk("openclaw")  # all drawers in the openclaw wing
    for d in drawers:
        print(d.tier, d.date, d.preview)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# Common stopwords to strip from filenames when extracting a "wing" name.
_WING_STOPWORDS = {
    "notes", "memory", "memories", "workspace",
    "the", "a", "an", "and", "or",
    "doc", "docs", "documents", "file", "files",
    "home", "user", "users", "tmp", "draft",
    "md", "txt", "text", "log", "logs",
}


def _wing_from_path(source_path: str) -> str:
    """Best-effort extraction of a wing name from a source path.

    Strategy:
      1. If the path ends in <slug>/<file>.md, use the slug.
      2. Else use the basename without extension.
      3. Strip stopwords; fall back to "<unknown>".

    Examples:
        /Users/me/notes/openclaw/2026-06-22.md  -> "openclaw"
        /home/x/projects/ai-py-boy/notes.md     -> "ai-py-boy"
        /Users/me/MEMORY.md                     -> "memory" (-> unknown after stop)
        /home/x/soul.md                          -> "soul" (-> unknown after stop)
    """
    if not source_path:
        return "<unknown>"
    p = Path(source_path)
    parts = p.parts
    # Try parent dir name first (most personal-note layouts).
    candidates: list[str] = []
    if len(parts) >= 2 and p.suffix.lower() in {".md", ".markdown", ".txt"}:
        candidates.append(parts[-2])
    # Then basename without extension.
    candidates.append(p.stem)
    for c in candidates:
        s = re.sub(r"[^a-z0-9]+", "-", c.lower()).strip("-")
        if not s:
            continue
        # Strip leading/trailing stopwords
        words = [w for w in s.split("-") if w and w not in _WING_STOPWORDS]
        if not words:
            continue
        return "-".join(words)
    return "<unknown>"


def _room_from_path(source_path: str, ingested_at: float) -> str:
    """Best-effort room (date) from a source path or ingest timestamp.

    Strategy:
      1. If the basename matches YYYY-MM-DD*.md, use that date.
      2. Else use the ingested_at date.
    """
    if source_path:
        m = re.search(r"(\d{4}-\d{2}-\d{2})", Path(source_path).stem)
        if m:
            return m.group(1)
    try:
        return datetime.fromtimestamp(ingested_at, tz=timezone.utc).date().isoformat()
    except (OSError, ValueError, OverflowError):
        return "1970-01-01"


@dataclass
class Drawer:
    """A single chunk in the palace."""
    chunk_id: str
    wing: str
    room: str
    tier: str
    preview: str          # first 200 chars
    source_path: str
    ingested_at: float
    importance: float = 0.0
    last_recalled_at: float = 0.0
    recall_count: int = 0
    verbatim: Optional[str] = None  # if source chunk has verbatim_text metadata

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "wing": self.wing,
            "room": self.room,
            "tier": self.tier,
            "preview": self.preview,
            "source_path": self.source_path,
            "ingested_at": self.ingested_at,
            "importance": self.importance,
            "last_recalled_at": self.last_recalled_at,
            "recall_count": self.recall_count,
            "verbatim": self.verbatim,
        }


@dataclass
class WingSummary:
    """Summary of one wing (person/project) in the palace."""
    name: str
    drawer_count: int = 0
    rooms: list[str] = field(default_factory=list)
    tiers: dict = field(default_factory=dict)  # tier -> count
    last_seen: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "drawer_count": self.drawer_count,
            "rooms": self.rooms,
            "tiers": self.tiers,
            "last_seen": self.last_seen,
        }


class PalaceIndex:
    """In-memory Wing/Room/Drawer index built from the brain store.

    The index is rebuilt on demand (cheap; one .get() per tier). For
    a 5000-chunk brain the index builds in ~1 second.
    """

    def __init__(self):
        self._drawers: list[Drawer] = []
        self._by_wing: dict[str, list[Drawer]] = {}

    @classmethod
    def from_store(cls, store) -> "PalaceIndex":
        """Build a palace index from a MemoryStore (backends.chroma)."""
        pi = cls()
        # Lazy import to avoid a hard dep cycle at module load.
        from .tier import Tier
        for tier_name in ("working", "episodic", "semantic", "procedural"):
            try:
                coll = store.collection_for(Tier(tier_name))
                data = coll.get(
                    limit=10000, include=["documents", "metadatas"],
                )
            except Exception:
                continue
            ids = (data or {}).get("ids") or []
            docs = (data or {}).get("documents") or []
            metas = (data or {}).get("metadatas") or []
            for cid, doc, md in zip(ids, docs, metas):
                md = md or {}
                if md.get("superseded_by"):
                    continue
                src = md.get("source_path", "") or ""
                ts = float(md.get("ingested_at") or 0.0)
                wing = _wing_from_path(src)
                room = _room_from_path(src, ts)
                preview = (doc or "")[:200].replace("\n", " ").replace("\r", " ")
                try:
                    imp = float(md.get("importance", 0.0))
                except (TypeError, ValueError):
                    imp = 0.0
                last_recall = float(md.get("last_recalled_at") or 0.0)
                recall_count = int(md.get("recall_count") or 0)
                verbatim = md.get("verbatim_text")
                d = Drawer(
                    chunk_id=cid,
                    wing=wing,
                    room=room,
                    tier=tier_name,
                    preview=preview,
                    source_path=src,
                    ingested_at=ts,
                    importance=imp,
                    last_recalled_at=last_recall,
                    recall_count=recall_count,
                    verbatim=verbatim,
                )
                pi._drawers.append(d)
                pi._by_wing.setdefault(wing, []).append(d)
        return pi

    def wings(self) -> list[WingSummary]:
        """Summary of every wing in the palace, sorted by last_seen desc."""
        out: list[WingSummary] = []
        for wing_name, drawers in self._by_wing.items():
            ws = WingSummary(name=wing_name, drawer_count=len(drawers))
            rooms: set[str] = set()
            tiers: dict[str, int] = {}
            last_seen = 0.0
            for d in drawers:
                rooms.add(d.room)
                tiers[d.tier] = tiers.get(d.tier, 0) + 1
                if d.ingested_at > last_seen:
                    last_seen = d.ingested_at
            ws.rooms = sorted(rooms, reverse=True)
            ws.tiers = tiers
            ws.last_seen = last_seen
            out.append(ws)
        out.sort(key=lambda w: w.last_seen, reverse=True)
        return out

    def walk(self, wing: str, room: Optional[str] = None,
             tier: Optional[str] = None, max_drawers: int = 100) -> list[Drawer]:
        """All drawers in a wing (optionally filtered to one room and/or
        one tier). Sorted by ingested_at desc."""
        drawers = self._by_wing.get(wing, [])
        if room is not None:
            drawers = [d for d in drawers if d.room == room]
        if tier is not None:
            drawers = [d for d in drawers if d.tier == tier]
        drawers.sort(key=lambda d: d.ingested_at, reverse=True)
        return drawers[:max_drawers]

    def all_drawers(self) -> list[Drawer]:
        """Flat list of all drawers, sorted by ingested_at desc."""
        return sorted(self._drawers, key=lambda d: d.ingested_at, reverse=True)
