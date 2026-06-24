"""
blocks.py — Memory blocks (Letta/MemGPT-inspired).

A "memory block" is a named, in-context chunk that the agent can read,
write, replace, and rethink. Think of it as a live, queryable version of
SOUL.md / AGENTS.md / USER.md — but stored as data, not as a file.

Examples of blocks:
  - "persona"     — who the agent is
  - "user"        — who the human is
  - "active_project" — what we're working on right now
  - "today_focus" — today's priorities
  - "rules"       — standing rules
  - "open_questions" — things to follow up on

Why blocks?
  - They're in-context: you can dump the whole block into the prompt
    without vector search.
  - They're self-editing: the agent (or cron, or watcher) can call
    block_replace() to update them.
  - They have history: every rewrite is logged with timestamp.
  - They're queryable: the graph + chroma can find related chunks.

Storage: SQLite (separate from graph.db, separate from chroma).
No LLM is required. This is a pure data structure.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


DEFAULT_BLOCKS_PATH = Path(__file__).resolve().parent.parent / "data" / "blocks.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS blocks (
    name        TEXT PRIMARY KEY,
    description TEXT,
    content     TEXT NOT NULL DEFAULT '',
    char_limit  INTEGER,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS block_history (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    old_content TEXT,
    new_content TEXT,
    operation   TEXT NOT NULL,   -- 'write' | 'replace' | 'rethink' | 'append'
    actor       TEXT,            -- who made the change ('agent' | 'cron' | 'user' | etc.)
    note        TEXT,            -- why
    created_at  REAL NOT NULL,
    FOREIGN KEY (name) REFERENCES blocks(name) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_block_history_name ON block_history(name);
CREATE INDEX IF NOT EXISTS idx_block_history_time ON block_history(created_at);
"""


@dataclass
class Block:
    name: str
    content: str
    description: Optional[str] = None
    char_limit: Optional[int] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "content": self.content,
            "description": self.description,
            "char_limit": self.char_limit,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class BlockStore:
    """In-context, self-editing memory blocks."""

    def __init__(self, path: str | Path = DEFAULT_BLOCKS_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # ---- CRUD --------------------------------------------------------------

    def get(self, name: str) -> Optional[Block]:
        row = self._conn.execute("SELECT * FROM blocks WHERE name = ?", (name,)).fetchone()
        if not row:
            return None
        return Block(
            name=row["name"],
            content=row["content"],
            description=row["description"],
            char_limit=row["char_limit"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def read(self, name: str) -> str:
        """Return block content, or empty string if not found."""
        b = self.get(name)
        return b.content if b else ""

    def create(self, name: str, content: str = "", description: Optional[str] = None,
               char_limit: Optional[int] = None,
               actor: str = "agent", note: Optional[str] = None) -> Block:
        """Create a new block. Errors if it already exists."""
        if self.get(name) is not None:
            raise ValueError(f"block '{name}' already exists; use write() or replace() to update")
        if char_limit is not None and len(content) > char_limit:
            raise ValueError(f"content length {len(content)} exceeds char_limit {char_limit}")
        now = time.time()
        self._conn.execute(
            "INSERT INTO blocks (name, description, content, char_limit, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (name, description, content, char_limit, now, now),
        )
        self._record_history(name, None, content, "create", actor, note)
        self._conn.commit()
        return Block(name=name, content=content, description=description,
                     char_limit=char_limit, created_at=now, updated_at=now)

    def write(self, name: str, content: str, actor: str = "agent",
              note: Optional[str] = None) -> Block:
        """Set block content (overwrites). Creates block if it doesn't exist.
        Respects char_limit."""
        existing = self.get(name)
        if existing is None:
            return self.create(name, content, actor=actor, note=note)
        if existing.char_limit is not None and len(content) > existing.char_limit:
            raise ValueError(
                f"content length {len(content)} exceeds char_limit {existing.char_limit}"
            )
        now = time.time()
        self._conn.execute(
            "UPDATE blocks SET content = ?, updated_at = ? WHERE name = ?",
            (content, now, name),
        )
        self._record_history(name, existing.content, content, "write", actor, note)
        self._conn.commit()
        return Block(name=name, content=content, description=existing.description,
                     char_limit=existing.char_limit,
                     created_at=existing.created_at, updated_at=now)

    def append(self, name: str, text: str, actor: str = "agent",
               note: Optional[str] = None) -> Block:
        """Append text to a block (with newline separator)."""
        existing = self.get(name)
        if existing is None:
            return self.create(name, text, actor=actor, note=note)
        new_content = (existing.content.rstrip("\n") + "\n" + text).strip("\n")
        return self.write(name, new_content, actor=actor, note=note)

    def replace(self, name: str, old: str, new: str, actor: str = "agent",
                note: Optional[str] = None) -> bool:
        """Replace the first occurrence of `old` with `new` in the block.
        Returns True if a replacement was made."""
        existing = self.get(name)
        if existing is None:
            raise ValueError(f"block '{name}' does not exist")
        if old not in existing.content:
            return False
        new_content = existing.content.replace(old, new, 1)
        self.write(name, new_content, actor=actor, note=note or f"replaced '{old[:40]}...'")
        return True

    def rethink(self, name: str, full_new_content: str, actor: str = "agent",
                note: Optional[str] = None) -> Block:
        """Replace the whole content with a re-thought version. Like Letta's
        `memory_rethink` tool — used when the agent re-evaluates a block."""
        return self.write(name, full_new_content, actor=actor,
                          note=note or "rethink")

    def delete(self, name: str) -> bool:
        cur = self._conn.execute("DELETE FROM blocks WHERE name = ?", (name,))
        self._conn.commit()
        return cur.rowcount > 0

    def list_blocks(self) -> list[Block]:
        rows = self._conn.execute("SELECT * FROM blocks ORDER BY name").fetchall()
        return [Block(
            name=r["name"], content=r["content"], description=r["description"],
            char_limit=r["char_limit"], created_at=r["created_at"], updated_at=r["updated_at"]
        ) for r in rows]

    def names(self) -> list[str]:
        return [r["name"] for r in self._conn.execute("SELECT name FROM blocks ORDER BY name")]

    # ---- History -----------------------------------------------------------

    def history(self, name: str, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM block_history WHERE name = ? ORDER BY created_at DESC LIMIT ?",
            (name, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def all_history(self, limit: int = 200) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM block_history ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- Stats -------------------------------------------------------------

    def stats(self) -> dict:
        n_blocks = self._conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0]
        n_writes = self._conn.execute("SELECT COUNT(*) FROM block_history").fetchone()[0]
        total_chars = self._conn.execute(
            "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM blocks"
        ).fetchone()[0]
        return {
            "blocks": n_blocks,
            "total_writes": n_writes,
            "total_chars": total_chars,
            "block_names": self.names(),
        }

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "BlockStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- Internals ---------------------------------------------------------

    def _record_history(self, name: str, old_content: Optional[str],
                        new_content: str, operation: str,
                        actor: Optional[str], note: Optional[str]) -> None:
        self._conn.execute(
            "INSERT INTO block_history (id, name, old_content, new_content, "
            "operation, actor, note, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), name, old_content, new_content, operation,
             actor, note, time.time()),
        )


# ---------------------------------------------------------------------------
# Common operations as a small API (so the agent and cron can call them)
# ---------------------------------------------------------------------------

def make_default_blocks(store: BlockStore) -> list[Block]:
    """Create the standard set of memory blocks if they don't exist."""
    defaults = [
        ("persona", "Who DuckBot is. Voice, identity, capabilities.",
         "🦆 DuckBot — personal AI agent for Ryan. Honest, capable, cost-conscious, cloud-only."),
        ("user", "Who Ryan/Duckets is. Preferences, birthday, context.",
         "Ryan (Duckets). Birthday: April 20. Prefers concise updates, rich Telegram messages, cloud-only models (no local LM Studio), strong on correctness > convenience."),
        ("active_project", "What we're working on right now.",
         "Upgrading the RAG/memory system (the 'large good brain' project) with temporal graph, entity extraction, memory blocks, and anti-injection scanner."),
        ("today_focus", "Top priorities for today.",
         "1) Complete the brain upgrade layers. 2) Rotate remaining API keys. 3) Verify memory system is clean."),
        ("open_questions", "Things to follow up on later.",
         "- Wire MCP server into OpenClaw config\n- Revert 31542dd1 (the 92-file git add -A mess)\n- MasterDashboard gateway plist"),
    ]
    created = []
    for name, desc, content in defaults:
        if store.get(name) is None:
            b = store.create(name, content, description=desc)
            created.append(b)
    return created
