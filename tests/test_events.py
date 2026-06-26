"""Tests for src/events.py — lifecycle event capture."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.events import (  # noqa: E402
    EventStore,
    SESSION_START,
    SESSION_END,
    PRE_TOOL_USE,
    POST_TOOL_USE,
    TOOL_ERROR,
    ALL_EVENT_TYPES,
)


@pytest.fixture
def store(tmp_path) -> EventStore:
    return EventStore(tmp_path / "events.db")


# ---------------------------------------------------------------------------
# Construction + schema
# ---------------------------------------------------------------------------


def test_eventstore_creates_db_and_schema(tmp_path):
    s = EventStore(tmp_path / "subdir" / "events.db")
    # Even with no writes, the schema should be applied (idempotent).
    stats = s.stats()
    assert stats["total"] == 0
    assert stats["by_type"] == {}
    assert (tmp_path / "subdir" / "events.db").exists()


def test_eventstore_idempotent_schema(tmp_path):
    """Multiple instances on the same path don't fail or duplicate tables."""
    p = tmp_path / "events.db"
    EventStore(p)
    EventStore(p)
    s = EventStore(p)
    s.record_event("s1", SESSION_START)
    assert s.stats()["total"] == 1


# ---------------------------------------------------------------------------
# record_event — basic round-trip
# ---------------------------------------------------------------------------


def test_record_event_returns_row_id(store):
    rid = store.record_event("sess-1", SESSION_START)
    assert isinstance(rid, int)
    assert rid > 0


def test_record_event_rejects_unknown_type(store):
    with pytest.raises(ValueError, match="unknown event_type"):
        store.record_event("sess-1", "bogus_event")


def test_record_event_round_trip_minimal(store):
    """Only session_id + event_type — everything else nullable."""
    rid = store.record_event("sess-1", SESSION_START)
    events = store.list_events()
    assert len(events) == 1
    e = events[0]
    assert e["id"] == rid
    assert e["session_id"] == "sess-1"
    assert e["event_type"] == SESSION_START
    assert e["tool_name"] is None
    assert e["args_json"] is None
    assert e["result_json"] is None
    assert e["error"] is None
    assert e["duration_ms"] is None
    assert e["context"] is None


def test_record_event_with_all_fields(store):
    store.record_event(
        "sess-2",
        POST_TOOL_USE,
        tool_name="brain_recall",
        args={"query": "x", "k": 3},
        result={"results": [{"chunk_id": "c1", "text": "hi"}]},
        duration_ms=42,
        context={"platform": "hermes", "agent_id": "mavis"},
    )
    e = store.list_events()[0]
    assert e["tool_name"] == "brain_recall"
    # JSON columns should round-trip back to native Python types
    # (the store's `_row_to_dict` parses args_json -> `args`).
    assert e["args"] == {"query": "x", "k": 3}
    assert e["result"] == {"results": [{"chunk_id": "c1", "text": "hi"}]}
    assert e["duration_ms"] == 42
    assert e["context"] == {"platform": "hermes", "agent_id": "mavis"}


def test_record_event_truncates_oversized_payloads(store):
    """Args > 8192 chars get truncated with an ellipsis marker."""
    huge = "x" * 20000
    store.record_event("sess-3", PRE_TOOL_USE, tool_name="brain_remember", args={"text": huge})
    e = store.list_events()[0]
    assert "[truncated" in (e.get("args_json") or "")
    # The parsed `args` dict still has the truncated string.
    parsed = json.loads(e["args_json"])
    assert len(parsed["text"]) < 20000
    assert "[truncated" in parsed["text"]


def test_record_event_handles_non_serializable_args(store):
    """default=str kicks in for non-JSON-native values."""
    store.record_event("sess-4", PRE_TOOL_USE, args={"obj": object(), "set": {1, 2, 3}})
    e = store.list_events()[0]
    # Set gets coerced to a string (Python set is not JSON-serializable).
    assert e["args"]["set"] in ("{1, 2, 3}", "[1, 2, 3]")  # str repr varies


def test_record_event_explicit_timestamp(store):
    """Caller can pass a fixed timestamp for testing / replay."""
    ts = 1_700_000_000.0
    store.record_event("sess-5", SESSION_END, timestamp=ts, error="agent_exit")
    e = store.list_events()[0]
    assert e["timestamp"] == ts
    assert e["error"] == "agent_exit"


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def test_list_events_filters_by_session(store):
    store.record_event("a", SESSION_START)
    store.record_event("b", SESSION_START)
    store.record_event("a", SESSION_END)
    a_events = store.list_events(session_id="a")
    assert len(a_events) == 2
    assert all(e["session_id"] == "a" for e in a_events)


def test_list_events_filters_by_type(store):
    store.record_event("s1", SESSION_START)
    store.record_event("s1", PRE_TOOL_USE, tool_name="brain_recall")
    store.record_event("s1", POST_TOOL_USE, tool_name="brain_recall")
    pre = store.list_events(event_type=PRE_TOOL_USE)
    assert len(pre) == 1
    assert pre[0]["event_type"] == PRE_TOOL_USE


def test_list_events_filters_by_since(store):
    """`since` is an inclusive lower-bound timestamp."""
    store.record_event("s1", SESSION_START, timestamp=100.0)
    store.record_event("s1", SESSION_END,   timestamp=200.0)
    store.record_event("s2", SESSION_START, timestamp=300.0)
    recent = store.list_events(since=200.0)
    # 200.0 inclusive + 300.0 = 2 events
    assert len(recent) == 2
    timestamps = [e["timestamp"] for e in recent]
    assert all(t >= 200.0 for t in timestamps)


def test_list_events_newest_first(store):
    store.record_event("s1", SESSION_START, timestamp=10.0)
    store.record_event("s1", SESSION_END, timestamp=20.0)
    store.record_event("s1", PRE_TOOL_USE, timestamp=15.0, tool_name="x")
    events = store.list_events()
    timestamps = [e["timestamp"] for e in events]
    assert timestamps == sorted(timestamps, reverse=True)


def test_list_events_limit(store):
    for i in range(20):
        store.record_event(f"s{i}", SESSION_START, timestamp=float(i))
    assert len(store.list_events(limit=5)) == 5


def test_session_events_returns_session_view(store):
    for sess in ("a", "a", "b", "a"):
        store.record_event(sess, SESSION_START)
    a = store.session_events("a")
    assert len(a) == 3
    assert all(e["session_id"] == "a" for e in a)


def test_recent_events_returns_all_sessions(store):
    store.record_event("a", SESSION_START)
    store.record_event("b", SESSION_END)
    store.record_event("c", PRE_TOOL_USE, tool_name="x")
    assert len(store.recent_events()) == 3


# ---------------------------------------------------------------------------
# stats + prune
# ---------------------------------------------------------------------------


def test_stats_aggregate_by_type(store):
    store.record_event("s1", SESSION_START)
    store.record_event("s2", SESSION_START)
    store.record_event("s1", SESSION_END, error="clean")
    store.record_event("s1", PRE_TOOL_USE, tool_name="x")
    stats = store.stats()
    assert stats["total"] == 4
    assert stats["by_type"][SESSION_START] == 2
    assert stats["by_type"][SESSION_END] == 1
    assert stats["by_type"][PRE_TOOL_USE] == 1


def test_stats_empty_store(store):
    stats = store.stats()
    assert stats == {"total": 0, "by_type": {}}


def test_prune_older_than(store):
    store.record_event("s1", SESSION_START, timestamp=1.0)
    store.record_event("s2", SESSION_START, timestamp=100.0)
    store.record_event("s3", SESSION_START, timestamp=200.0)
    deleted = store.prune_older_than(days=1.0 / 86400.0)  # ~1 second ago
    # Everything older than (now - 1s) gets deleted; depends on `now`
    # but the rows with timestamps 1.0 and 100.0 are almost certainly gone.
    assert deleted >= 2
    assert store.stats()["total"] <= 1


def test_all_event_types_constant():
    """Sanity: ALL_EVENT_TYPES matches the constants."""
    assert ALL_EVENT_TYPES == frozenset({
        SESSION_START, SESSION_END, PRE_TOOL_USE, POST_TOOL_USE, TOOL_ERROR,
    })


# ---------------------------------------------------------------------------
# Concurrent writes (lock)
# ---------------------------------------------------------------------------


def test_concurrent_writes_dont_lose_rows(tmp_path):
    """Multiple threads writing concurrently — every event must land."""
    import threading

    s = EventStore(tmp_path / "events.db")
    N = 50
    errors: list[Exception] = []

    def worker(start: int) -> None:
        try:
            for i in range(start, start + N):
                s.record_event(f"sess-{i}", SESSION_START, context={"i": i})
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i * N,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert s.stats()["total"] == 4 * N


# ---------------------------------------------------------------------------
# Integration: actual MCP-style use case
# ---------------------------------------------------------------------------


def test_session_lifecycle_simulation(store):
    """Walk a realistic session: start → tool call → tool error → end."""
    store.record_event("agent-1", SESSION_START, context={"platform": "openclaw"})
    store.record_event(
        "agent-1", PRE_TOOL_USE,
        tool_name="brain_recall",
        args={"query": "cloud models", "k": 5},
    )
    store.record_event(
        "agent-1", POST_TOOL_USE,
        tool_name="brain_recall",
        result={"results": [{"chunk_id": "c1", "score": 0.7}]},
        duration_ms=42,
    )
    store.record_event(
        "agent-1", TOOL_ERROR,
        tool_name="brain_palace",
        error="ChromaDB native segfault",
    )
    store.record_event("agent-1", SESSION_END, error="clean")

    trace = store.session_events("agent-1")
    assert [e["event_type"] for e in trace] == [
        SESSION_END, TOOL_ERROR, POST_TOOL_USE, PRE_TOOL_USE, SESSION_START,
    ]  # newest-first

    # The tool_error carries the segfault message — operator can grep it.
    err = next(e for e in trace if e["event_type"] == TOOL_ERROR)
    assert "ChromaDB native segfault" in err["error"]