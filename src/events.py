"""events.py — lifecycle event capture for tool/session auditing.

Inspired by MindBank's "lifecycle event capture" (pre_tool_use /
post_tool_use / stop). Re-implemented natively against duckbot's
existing SQLite infrastructure.

Captured event types:
    session_start      — agent session begins
    session_end        — agent session ends
    pre_tool_use       — before a brain tool is called (args captured)
    post_tool_use      — after a brain tool returns (result + duration captured)
    tool_error         — tool raised an exception (error message captured)

Schema (SQLite, one table in data/events.db):
    id            INTEGER PRIMARY KEY AUTOINCREMENT
    session_id    TEXT NOT NULL        — agent session (UUID/ULID)
    event_type    TEXT NOT NULL        — one of the names above
    timestamp     REAL NOT NULL        — Unix epoch seconds
    tool_name     TEXT                — set for pre/post/tool_error
    args_json     TEXT                — set for pre_tool_use
    result_json   TEXT                — set for post_tool_use (truncated)
    error         TEXT                — set for tool_error / session_end reason
    duration_ms   INTEGER             — set for post_tool_use
    context       TEXT                — free-form context (e.g. platform, agent_id)

Indexed on (session_id, timestamp) so a session's events are cheap
to walk in order.

Public API:
    EventStore(path=...) — SQLite-backed store, context-managed
    record_event(...)    — fire-and-forget write (sync, fast)
    list_events(...)     — read API for "what happened in session X"
    recent_events(...)   — tail of all events across sessions
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


# Event type constants. Exported so callers don't sprinkle string literals.
SESSION_START = "session_start"
SESSION_END = "session_end"
PRE_TOOL_USE = "pre_tool_use"
POST_TOOL_USE = "post_tool_use"
TOOL_ERROR = "tool_error"

ALL_EVENT_TYPES = frozenset({
    SESSION_START,
    SESSION_END,
    PRE_TOOL_USE,
    POST_TOOL_USE,
    TOOL_ERROR,
})

# Soft cap on per-payload length (chars in the serialized JSON).
# Anything longer gets truncated BEFORE serialization so we never
# produce malformed JSON (which would break all reads of that row).
# Truncation is recursive for dict/list values — only the oversized
# leaf strings get the ellipsis marker; small keys/values stay intact.
_MAX_PAYLOAD_CHARS = 8192
_TRUNCATION_MARKER = "...[truncated]"


def _truncate_value(v: Any, budget: int = _MAX_PAYLOAD_CHARS) -> tuple[Any, bool]:
    """Recursively shrink `v` until its JSON repr fits in `budget` chars.

    Returns (possibly-truncated value, was_truncated_flag).
    """
    if isinstance(v, str):
        if len(v) > budget:
            return v[:budget] + _TRUNCATION_MARKER, True
        return v, False
    if isinstance(v, (int, float, bool)) or v is None:
        return v, False
    if isinstance(v, dict):
        # Truncate values one at a time, biggest first, until we fit.
        items = sorted(v.items(), key=lambda kv: -len(json.dumps(kv[1], default=str)))
        out = {}
        truncated_any = False
        for k, val in items:
            shrunk, was = _truncate_value(val, budget // max(1, len(items)))
            out[k] = shrunk
            truncated_any = truncated_any or was
            if was:
                # Once we've truncated one leaf, re-serialize and check.
                if len(json.dumps(out, default=str)) > budget:
                    # Drop remaining items rather than emit broken JSON.
                    out[k] = _TRUNCATION_MARKER
                if len(json.dumps(out, default=str)) <= budget:
                    break
        return out, truncated_any
    if isinstance(v, list):
        out_list = []
        truncated_any = False
        for item in v:
            shrunk, was = _truncate_value(item, budget // max(1, len(v)))
            out_list.append(shrunk)
            truncated_any = truncated_any or was
            if len(json.dumps(out_list, default=str)) > budget:
                out_list.append(_TRUNCATION_MARKER)
                break
        return out_list, truncated_any
    # Other types (set, custom objects) — coerce to str + truncate.
    s = str(v)
    if len(s) > budget:
        return s[:budget] + _TRUNCATION_MARKER, True
    return v, False


def _safe_json(v: Any, budget: int = _MAX_PAYLOAD_CHARS) -> Optional[str]:
    """Serialize `v` to JSON, recursively truncating oversized leaf strings.

    Returns None if `v` is None.
    """
    if v is None:
        return None
    shrunk, _ = _truncate_value(v, budget)
    return json.dumps(shrunk, default=str)


class EventStore:
    """SQLite-backed lifecycle event log.

    Thread-safe via a per-instance lock (the MCP server is single-
    threaded asyncio, but the watcher + Hermes plugin can write
    concurrently).
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS events (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id    TEXT NOT NULL,
        event_type    TEXT NOT NULL,
        timestamp     REAL NOT NULL,
        tool_name     TEXT,
        args_json     TEXT,
        result_json   TEXT,
        error         TEXT,
        duration_ms   INTEGER,
        context       TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_events_session_ts
        ON events (session_id, timestamp);
    CREATE INDEX IF NOT EXISTS idx_events_type_ts
        ON events (event_type, timestamp);
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.path))
        conn.executescript(self.SCHEMA)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def record_event(
        self,
        session_id: str,
        event_type: str,
        *,
        tool_name: Optional[str] = None,
        args: Any = None,
        result: Any = None,
        error: Optional[str] = None,
        duration_ms: Optional[int] = None,
        context: Optional[dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> int:
        """Fire-and-forget event write. Returns the inserted row id.

        Sync — keep this fast. The callers (MCP handlers, plugin hooks)
        are already on async paths, so a sync SQLite write inside a
        quick lock is fine. Don't await anything from here.
        """
        if event_type not in ALL_EVENT_TYPES:
            raise ValueError(
                f"unknown event_type {event_type!r}; expected one of {sorted(ALL_EVENT_TYPES)}"
            )
        args_json = _safe_json(args) if args is not None else None
        result_json = _safe_json(result) if result is not None else None
        ctx_json = _safe_json(context) if context is not None else None
        ts = timestamp if timestamp is not None else time.time()

        with self._lock, self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO events
                    (session_id, event_type, timestamp, tool_name,
                     args_json, result_json, error, duration_ms, context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    event_type,
                    ts,
                    tool_name,
                    args_json,
                    result_json,
                    error,
                    duration_ms,
                    ctx_json,
                ),
            )
            return int(cur.lastrowid or 0)

    def list_events(
        self,
        session_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100,
        since: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """Read events, newest-first. Optional filters by session / type / timestamp."""
        clauses: list[str] = []
        params: list[Any] = []
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if event_type is not None:
            clauses.append("event_type = ?")
            params.append(event_type)
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, int(limit)))

        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM events {where} ORDER BY timestamp DESC LIMIT ?",
                params,
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        """Tail of all events across sessions, newest-first."""
        return self.list_events(limit=limit)

    def session_events(self, session_id: str, limit: int = 200) -> list[dict[str, Any]]:
        """All events for a single session, newest-first."""
        return self.list_events(session_id=session_id, limit=limit)

    def stats(self) -> dict[str, Any]:
        """Aggregate counts by event_type."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT event_type, COUNT(*) as n FROM events GROUP BY event_type"
            ).fetchall()
            total_row = conn.execute("SELECT COUNT(*) as n FROM events").fetchone()
        by_type = {r["event_type"]: r["n"] for r in rows}
        return {
            "total": total_row["n"] if total_row else 0,
            "by_type": by_type,
        }

    def prune_older_than(self, days: float) -> int:
        """Delete events older than `days` days. Returns rows deleted."""
        cutoff = time.time() - days * 86400.0
        with self._lock, self._conn() as conn:
            cur = conn.execute("DELETE FROM events WHERE timestamp < ?", (cutoff,))
            return cur.rowcount or 0

    @staticmethod
    def _row_to_dict(r: sqlite3.Row) -> dict[str, Any]:
        d = dict(r)
        # Parse JSON columns back to Python objects so callers get native
        # types (dict / list / str / int / float / None).
        for col in ("args_json", "result_json", "context"):
            if d.get(col):
                try:
                    d[col.rsplit("_json", 1)[0]] = json.loads(d[col])
                except (json.JSONDecodeError, ValueError):
                    pass
        return d


__all__ = [
    "EventStore",
    "SESSION_START",
    "SESSION_END",
    "PRE_TOOL_USE",
    "POST_TOOL_USE",
    "TOOL_ERROR",
    "ALL_EVENT_TYPES",
]