"""
test_blocks.py — tests for memory blocks (Layer 3).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.blocks import BlockStore, make_default_blocks


@pytest.fixture
def store(tmp_path):
    s = BlockStore(path=tmp_path / "blocks.db")
    yield s
    s.close()


# --- CRUD -------------------------------------------------------------------

def test_create_block(store):
    b = store.create("persona", "DuckBot is great.", description="Who I am")
    assert b.name == "persona"
    assert b.content == "DuckBot is great."
    assert b.description == "Who I am"
    assert b.created_at > 0


def test_create_duplicate_raises(store):
    store.create("foo", "bar")
    with pytest.raises(ValueError, match="already exists"):
        store.create("foo", "baz")


def test_get_block(store):
    store.create("test", "hello")
    b = store.get("test")
    assert b is not None
    assert b.content == "hello"


def test_get_nonexistent_returns_none(store):
    assert store.get("nope") is None


def test_read_returns_empty_string_for_missing(store):
    assert store.read("nope") == ""


def test_write_creates_if_missing(store):
    b = store.write("new_block", "first content")
    assert b.content == "first content"
    assert store.get("new_block") is not None


def test_write_overwrites_existing(store):
    store.create("foo", "original")
    b = store.write("foo", "updated")
    assert b.content == "updated"
    assert store.get("foo").content == "updated"


def test_append_to_existing(store):
    store.create("notes", "line 1")
    b = store.append("notes", "line 2")
    assert "line 1" in b.content
    assert "line 2" in b.content


def test_append_creates_if_missing(store):
    b = store.append("new", "first line")
    assert b.content == "first line"


def test_replace_first_occurrence(store):
    store.create("rules", "No local models. No local models. No local models.")
    changed = store.replace("rules", "No local models", "Cloud-only")
    assert changed
    assert "Cloud-only" in store.read("rules")
    # First occurrence was replaced; two remain
    assert store.read("rules").count("No local models") == 2


def test_replace_returns_false_when_not_found(store):
    store.create("rules", "something else")
    changed = store.replace("rules", "missing", "replacement")
    assert not changed


def test_rethink_replaces_whole_content(store):
    store.create("persona", "Old version")
    b = store.rethink("persona", "New version after reflection")
    assert b.content == "New version after reflection"
    assert store.read("persona") == "New version after reflection"


def test_delete_block(store):
    store.create("temp", "data")
    assert store.delete("temp")
    assert store.get("temp") is None
    assert not store.delete("temp")  # second delete is a no-op


# --- Char limits ------------------------------------------------------------

def test_char_limit_enforced_on_create(store):
    with pytest.raises(ValueError, match="exceeds char_limit"):
        store.create("small", "x" * 100, char_limit=10)


def test_char_limit_enforced_on_write(store):
    store.create("small", "hi", char_limit=10)
    with pytest.raises(ValueError, match="exceeds char_limit"):
        store.write("small", "x" * 50)


# --- History ----------------------------------------------------------------

def test_history_recorded(store):
    store.create("foo", "v1")
    store.write("foo", "v2")
    store.write("foo", "v3")
    hist = store.history("foo")
    assert len(hist) == 3
    # Newest first
    assert hist[0]["new_content"] == "v3"
    assert hist[0]["old_content"] == "v2"
    assert hist[0]["operation"] == "write"


def test_history_includes_actor_and_note(store):
    store.create("foo", "v1", actor="cron", note="initial seed")
    hist = store.history("foo")
    assert hist[0]["actor"] == "cron"
    assert hist[0]["note"] == "initial seed"


def test_history_limit(store):
    for i in range(10):
        store.write("counter", f"v{i}")
    hist = store.history("counter", limit=3)
    assert len(hist) == 3


def test_cascade_delete_history(store):
    store.create("temp", "x")
    store.write("temp", "y")
    store.delete("temp")
    assert store.history("temp") == []


# --- Listing ----------------------------------------------------------------

def test_list_blocks(store):
    store.create("a", "1")
    store.create("b", "2")
    store.create("c", "3")
    blocks = store.list_blocks()
    assert {b.name for b in blocks} == {"a", "b", "c"}


def test_names(store):
    store.create("alpha", "x")
    store.create("beta", "y")
    assert store.names() == ["alpha", "beta"]


# --- Stats ------------------------------------------------------------------

def test_stats(store):
    store.create("a", "hello", char_limit=100)
    store.create("b", "world!")
    s = store.stats()
    assert s["blocks"] == 2
    assert s["total_writes"] == 2
    assert s["total_chars"] == 11  # 5 + 6
    assert set(s["block_names"]) == {"a", "b"}


# --- Defaults ---------------------------------------------------------------

def test_make_default_blocks(store):
    created = make_default_blocks(store)
    assert len(created) >= 4  # persona, user, active_project, today_focus, open_questions
    assert store.get("persona") is not None
    assert store.get("user") is not None


def test_make_default_blocks_idempotent(store):
    make_default_blocks(store)
    second = make_default_blocks(store)
    assert second == [], "second call should be a no-op"


# --- Context manager --------------------------------------------------------

def test_context_manager(tmp_path):
    with BlockStore(path=tmp_path / "ctx.db") as s:
        s.create("x", "y")
        assert s.get("x") is not None
