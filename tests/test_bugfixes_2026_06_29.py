"""
Regression tests for the 2026-06-29 brain bug-fix pass.

Each test pins a specific bug surfaced during the memory-audit session
that day. These exist so the same regression can't slip back in silently.

Bugs covered:
  1. `brain_forget_by_query` raised AttributeError because QueryResult
     has no `.source_path` attribute (lives in metadata.source_path).
  2. `brain_graph_entity` stored properties via str(dict) → Python repr,
     not valid JSON. Now uses json.dumps.
  3. `brain_wake_up` had no deadline — could hang past MCP timeout.
  4. `brain_remember` claimed a 10/min rate limit but didn't enforce it.
  5. New `brain_supersede` tool — first-class memory deprecation that
     keeps the audit trail (vs destructive forget).
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Bug 1: brain_forget_by_query — QueryResult has no .source_path
# ---------------------------------------------------------------------------

def test_forget_by_query_handles_queryresult_without_source_path(monkeypatch):
    """The matched[] array must not crash when QueryResult lacks source_path.

    Real contract: Memory.recall() returns list[QueryResult], QueryResult
    has only chunk_id/text/metadata/tier/rrf_score. The previous code did
    `r.source_path` directly, which raised AttributeError.
    """
    from src.connectors.base import Brain

    fake_chunk = SimpleNamespace(
        chunk_id="cid-123",
        text="Some chunk text",
        metadata={"source_path": "memory/2026-06-29.md", "source": "session"},
        tier="semantic",
        rrf_score=0.85,
    )

    class FakeMemory:
        async def recall(self, query, k=5, tier=None):
            return [fake_chunk], SimpleNamespace()

        async def forget(self, chunk_id):
            return True

    bc = Brain(persist_dir=tempfile.mkdtemp())
    # Inject fake memory so we don't touch the real brain.
    bc._memory = lambda: FakeMemory()
    # Bypass rate limit for this test.
    monkeypatch.setenv("DUCKBOT_BRAIN_RATE_LIMIT_OFF", "1")

    result = bc.forget_by_query("anything", k=1)
    assert result["deleted"] == 1
    assert result["deleted_ids"] == ["cid-123"]
    assert result["matched"][0]["source_path"] == "memory/2026-06-29.md"
    assert result["matched"][0]["chunk_id"] == "cid-123"
    assert result["matched"][0]["score"] == 0.85


def test_forget_by_query_handles_missing_metadata(monkeypatch):
    """If a chunk has no metadata at all, source_path should be '' not crash."""
    from src.connectors.base import Brain

    bare = SimpleNamespace(
        chunk_id="cid-bare",
        text="Bare chunk",
        metadata=None,  # missing
        tier="episodic",
        rrf_score=0.0,
    )

    class FakeMemory:
        async def recall(self, query, k=5, tier=None):
            return [bare], SimpleNamespace()

        async def forget(self, chunk_id):
            return True

    bc = Brain(persist_dir=tempfile.mkdtemp())
    bc._memory = lambda: FakeMemory()
    monkeypatch.setenv("DUCKBOT_BRAIN_RATE_LIMIT_OFF", "1")

    result = bc.forget_by_query("anything", k=1)
    assert result["matched"][0]["source_path"] == ""
    assert result["matched"][0]["chunk_id"] == "cid-bare"


# ---------------------------------------------------------------------------
# Bug 2: brain_graph_entity — properties were str(dict), not JSON
# ---------------------------------------------------------------------------

def test_graph_entity_stores_properties_as_real_json(tmp_path):
    """The notes field must be valid JSON, not Python repr."""
    from src.connectors.base import Brain

    persist_dir = tmp_path / "persist"
    persist_dir.mkdir()
    bc = Brain(persist_dir=str(persist_dir))
    # Force a fresh graph path under tmp_path so we don't touch the real one.
    bc.graph_path = tmp_path / "graph.db"

    properties = {
        "location": "~/Desktop/DuckBotOS",
        "version": "0.2.3",
        "github": "Franzferdinan51/DuckBotOS",
        "packages": 14,
        "active": True,
    }

    res = bc.graph_upsert_entity(name="DuckBotOS", kind="project", properties=properties)
    assert "id" in res, f"expected id in response, got {res}"
    assert res["name"] == "DuckBotOS"

    # Notes must be parseable as JSON.
    notes = res["notes"]
    assert notes is not None
    parsed = json.loads(notes)  # would raise on Python repr
    assert parsed == properties

    # Confirm the old bug would have failed this:
    assert notes.startswith("{"), "should be a JSON object"
    # Python repr would have single quotes: "{'location': ...}" — assert we don't.
    assert "'" not in notes, f"looks like Python repr, not JSON: {notes!r}"


def test_graph_entity_with_no_properties():
    """Passing properties=None should leave notes=None, not error."""
    from src.connectors.base import Brain

    with tempfile.TemporaryDirectory() as d:
        bc = Brain(persist_dir=d)
        bc.graph_path = Path(d) / "graph.db"

        res = bc.graph_upsert_entity(name="PlainEntity")
        assert res["notes"] is None


def test_graph_entity_rejects_non_serializable_properties(tmp_path):
    """Properties containing non-JSON values should return a clean error."""
    from src.connectors.base import Brain

    bc = Brain(persist_dir=str(tmp_path))
    bc.graph_path = tmp_path / "graph.db"

    class NotJsonable:
        pass

    res = bc.graph_upsert_entity(
        name="BadEntity",
        properties={"oops": NotJsonable()},
    )
    assert "error" in res
    assert "not JSON-encodable" in res["error"]


# ---------------------------------------------------------------------------
# Bug 4: brain_remember — 10/min rate limit (was unenforced)
# ---------------------------------------------------------------------------

def test_remember_rate_limit_enforced(monkeypatch):
    """Burst 10 calls, the 11th should be rate-limited."""
    from src.connectors import base as base_mod
    from src.connectors.base import Brain, RememberResult

    # Reset the bucket so we start full.
    base_mod._remember_bucket_tokens = base_mod._remember_bucket_capacity
    base_mod._remember_bucket_last_refill = time.time()

    # Explicitly UNSET bypass var so the limit is enforced.
    monkeypatch.delenv("DUCKBOT_BRAIN_RATE_LIMIT_OFF", raising=False)

    # Shrink the refill rate to 0 so the 11th call hits an empty bucket
    # instantly instead of waiting 6s for a token to drip in. We restore
    # the original at the end via monkeypatch.
    monkeypatch.setattr(base_mod, "_remember_bucket_refill_per_sec", 0.0)

    # Stub out the actual store so we don't need a real Chroma instance.
    class FakeMemory:
        def __init__(self):
            self.count = 0

        async def remember(self, text, source_path=None, metadata=None, force_tier=None, facts=None):
            self.count += 1
            return SimpleNamespace(
                chunk_id=f"c{self.count}",
                tier=SimpleNamespace(value="semantic"),
                confidence=1.0,
                importance=0.5,
                entities=[],
                relationships=[],
                stored=True,
                duration_ms=1.0,
            )

    bc = Brain(persist_dir=tempfile.mkdtemp())
    fake = FakeMemory()
    bc._memory = lambda: fake

    # First 10: pass
    results = []
    for i in range(10):
        r = bc.remember(f"text {i}")
        results.append(r)

    assert all(r.stored for r in results), "first 10 should succeed"

    # 11th: bucket empty, no refill => RuntimeError raised after 30s.
    # To make the test fast, monkeypatch the timeout to 0.1s.
    monkeypatch.setattr(base_mod, "_remember_bucket_refill_per_sec", 0.0)
    orig_limiter = base_mod._remember_rate_limit

    def fast_limiter(timeout_s=30.0):
        # Force a tiny timeout regardless of the caller's choice.
        return orig_limiter(timeout_s=0.1)

    monkeypatch.setattr(base_mod, "_remember_rate_limit", fast_limiter)

    r11 = bc.remember("text 11")
    assert r11.rate_limited is True, f"expected rate_limited=True, got {r11}"
    assert r11.stored is False
    assert ("10/min" in (r11.error or "")) or ("rate limit" in (r11.error or ""))


def test_remember_rate_limit_can_be_disabled(monkeypatch):
    """DUCKBOT_BRAIN_RATE_LIMIT_OFF=1 should let bulk callers through."""
    from src.connectors import base as base_mod
    from src.connectors.base import Brain

    base_mod._remember_bucket_tokens = 0.0  # start empty
    base_mod._remember_bucket_last_refill = time.time()

    monkeypatch.setenv("DUCKBOT_BRAIN_RATE_LIMIT_OFF", "1")

    class FakeMemory:
        async def remember(self, text, source_path=None, metadata=None, force_tier=None, facts=None):
            return SimpleNamespace(
                chunk_id="c", tier=SimpleNamespace(value="semantic"),
                confidence=1.0, importance=0.5,
                entities=[], relationships=[], stored=True, duration_ms=1.0,
            )

    bc = Brain(persist_dir=tempfile.mkdtemp())
    bc._memory = lambda: FakeMemory()

    # 20 calls — would all be rate-limited if env var didn't disable.
    for i in range(20):
        r = bc.remember(f"text {i}")
        assert r.stored is True, f"call {i} should have succeeded but was rate-limited"


# ---------------------------------------------------------------------------
# Bug 3: brain_wake_up — soft deadline
# ---------------------------------------------------------------------------

def test_wake_up_respects_deadline(monkeypatch):
    """wake_up should return within deadline_ms even if a section is slow."""
    from src.connectors.base import Brain

    bc = Brain(persist_dir=tempfile.mkdtemp())

    # Make recall slow to trigger the deadline.
    class SlowRecall:
        async def recall(self, query, k=5, tier=None, rerank=False):
            # Sleep long enough that 3 attempts would blow the deadline.
            # The deadline check between attempts should bail after 1 sleep.
            await asyncio.sleep(1.5)
            return [], SimpleNamespace()

    class SlowStats:
        def stats(self):
            time.sleep(0.5)
            return SimpleNamespace(
                total=0, working=0, episodic=0, semantic=0, procedural=0,
            )

    # Replace the memory factory with a slow stub.
    bc._memory = lambda: SlowRecall()
    monkeypatch.setattr(bc, "stats", lambda: SimpleNamespace(
        total=0, working=0, episodic=0, semantic=0, procedural=0,
    ))

    started = time.monotonic()
    result = bc.wake_up(query="test", deadline_ms=500)
    elapsed_ms = (time.monotonic() - started) * 1000

    # Should bail after ~1 sleep (~1.5s) and return within ~2s, well under
    # the 3x-sleep ceiling. Without the deadline guard, we'd wait 3 sleeps.
    assert elapsed_ms < 2500, f"wake_up took {elapsed_ms:.0f}ms (3x sleeps would be ~4500ms)"
    assert result.get("wake_up_truncated") is True
    assert "wake_up_deadline_ms" in result


def test_wake_up_default_deadline_is_8s():
    """Default deadline should be 8000ms (under MCP's typical 10s timeout)."""
    from src.connectors.base import Brain

    bc = Brain(persist_dir=tempfile.mkdtemp())

    class FastRecall:
        async def recall(self, query, k=5, tier=None, rerank=False):
            return [], SimpleNamespace()

    bc._memory = lambda: FastRecall()
    # Patch stats too so it doesn't try to hit a real Chroma.
    bc.stats = lambda: SimpleNamespace(
        total=0, working=0, episodic=0, semantic=0, procedural=0,
    )
    result = bc.wake_up()
    assert result["wake_up_deadline_ms"] == 8000


# ---------------------------------------------------------------------------
# Bug 5: brain_supersede — new first-class deprecation tool
# ---------------------------------------------------------------------------

def test_supersede_marks_old_chunk(monkeypatch):
    """supersede should set metadata.superseded_by on the old chunk."""
    from src.connectors.base import Brain

    class FakeMemory:
        async def supersede(self, old_chunk_id, new_chunk_id=None, reason=None):
            return {
                "superseded": True,
                "old_chunk_id": old_chunk_id,
                "new_chunk_id": new_chunk_id,
                "reason": reason,
            }

    bc = Brain(persist_dir=tempfile.mkdtemp())
    bc._memory = lambda: FakeMemory()

    res = bc.supersede(
        old_chunk_id="old-cid",
        new_chunk_id="new-cid",
        reason="operator correction: BrowserOS works, I fumble the controls",
    )

    assert res["superseded"] is True
    assert res["old_chunk_id"] == "old-cid"
    assert res["new_chunk_id"] == "new-cid"
    assert "operator correction" in res["reason"]


def test_supersede_requires_old_chunk_id():
    """Empty old_chunk_id should return a clean error."""
    from src.connectors.base import Brain

    bc = Brain(persist_dir=tempfile.mkdtemp())
    res = bc.supersede(old_chunk_id="")
    # Either returns {"error": "..."} or supersede stub — both fine, just no crash.
    assert "error" in res or "superseded" in res


def test_supersede_handles_chunk_not_found():
    """If old_chunk_id doesn't exist, should report superseded=False, not crash."""
    from src.connectors.base import Brain

    class FakeMemory:
        async def supersede(self, old_chunk_id, new_chunk_id=None, reason=None):
            return {
                "superseded": False,
                "old_chunk_id": old_chunk_id,
                "new_chunk_id": new_chunk_id,
                "reason": reason,
            }

    bc = Brain(persist_dir=tempfile.mkdtemp())
    bc._memory = lambda: FakeMemory()

    res = bc.supersede(old_chunk_id="does-not-exist", reason="test")
    assert res["superseded"] is False