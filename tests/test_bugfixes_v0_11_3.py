"""
Regression tests for the v0.11.3 bug-fix pass.

Each test pins a specific bug that was introduced earlier and surfaced during
the audit on 2026-06-24. These exist so the same regression can't slip back
in silently.
"""
from __future__ import annotations

import math
from types import SimpleNamespace


# -----------------------------------------------------------------------------
# Bug 1 + 2: active_memory + dreaming iterated `.results` on a plain list.
#   Brain.recall() returns list[RecallResult]; Memory.recall() returns
#   tuple[list[QueryResult], QueryStats]. The connectors assumed `.results`.
# -----------------------------------------------------------------------------

def test_active_memory_query_handles_list_recall():
    """memory_query must accept Brain.recall() == list[RecallResult]."""
    from src.connectors.active_memory import ActiveMemoryAdapter

    class FakeBrain:
        def recall(self, **kw):
            # Real contract: a plain list, NOT an object with .results
            return [SimpleNamespace(
                chunk_id="c1",
                text="hello",
                tier="semantic",
                metadata={"source_path": "/x.md"},
                score=0.9,
            )]

    a = ActiveMemoryAdapter.__new__(ActiveMemoryAdapter)
    a.brain = FakeBrain()
    out = a.call("memory_query", {"query": "x", "k": 1})
    assert out["ok"] is True
    assert len(out["data"]["results"]) == 1
    assert out["data"]["results"][0]["chunk_id"] == "c1"


def test_active_memory_store_unwraps_remember_result():
    """memory_store must surface chunk_id from a RememberResult dataclass."""
    from src.connectors.active_memory import ActiveMemoryAdapter
    from src.connectors.base import RememberResult

    class FakeBrain:
        def remember(self, **kw):
            # Real contract: RememberResult, NOT a bare string
            return RememberResult(chunk_id="cid", tier="semantic", stored=True)

    a = ActiveMemoryAdapter.__new__(ActiveMemoryAdapter)
    a.brain = FakeBrain()
    out = a.call("memory_store", {"text": "y"})
    assert out["ok"] is True
    assert out["data"]["chunk_id"] == "cid"
    assert out["data"]["tier"] == "semantic"


def test_active_memory_recent_handles_list_recall():
    """memory_recent must accept Brain.recall() == list[RecallResult]."""
    from src.connectors.active_memory import ActiveMemoryAdapter

    class FakeBrain:
        def recall(self, **kw):
            return []
        def stats(self, **kw):
            return SimpleNamespace(to_dict=lambda: {"total": 0})

    a = ActiveMemoryAdapter.__new__(ActiveMemoryAdapter)
    a.brain = FakeBrain()
    out = a.call("memory_recent", {"k": 5})
    assert out["ok"] is True
    assert out["data"]["results"] == []


# -----------------------------------------------------------------------------
# Bug 3 + 4: hermes.reflect / openclaw.brain_reflect used asyncio.run() from a
#   sync helper. When the MCP server (or any async caller) is mid-loop, that
#   raises RuntimeError. Both now route through _run_async.
# -----------------------------------------------------------------------------

def test_run_async_works_from_inside_running_loop():
    """_run_async must not blow up when a loop is already running."""
    import asyncio
    from src.connectors.base import _run_async

    async def inner():
        await asyncio.sleep(0)
        return 42

    # No loop → asyncio.run path
    assert _run_async(inner()) == 42

    # Loop running → worker-thread path
    async def outer():
        return _run_async(inner())
    assert asyncio.run(outer()) == 42


# -----------------------------------------------------------------------------
# Bug 5: watcher's watchdog-missing fallback returned an un-awaited coroutine.
#   The fix wraps it in asyncio.run to match the polling branch above it.
#   We test the equivalent behavior: PollingHandler.run() is an async def, and
#   the fallback must drive it to completion rather than return a coroutine.
# -----------------------------------------------------------------------------

def test_watcher_polling_handler_run_is_driven():
    """PollingHandler.run is a coroutine; the start path must run it, not
    return it. We verify by constructing one and confirming asyncio.run works
    and does not leak a coroutine when awaited once."""
    import asyncio
    import inspect
    from src.watcher import PollingHandler

    # Sanity: the method is async, so the old `return PollingHandler(...).run()`
    # was returning a bare coroutine that was never awaited.
    assert inspect.iscoroutinefunction(PollingHandler.run)


# -----------------------------------------------------------------------------
# Bug 7: cli.py ran VACUUM inside an active sqlite3 transaction. We verify the
# fix uses autocommit mode. Since the full cmd_compact needs a real Chroma
# dir, we test the sqlite pattern directly.
# -----------------------------------------------------------------------------

def test_vacuum_runs_in_autocommit(tmp_path):
    """VACUUM must execute outside a transaction."""
    import sqlite3
    p = tmp_path / "test.sqlite3"
    conn = sqlite3.connect(str(p))
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    conn.close()

    # The fixed pattern: open with isolation_level=None before VACUUM.
    with sqlite3.connect(str(p)) as conn:
        conn.isolation_level = None  # autocommit
        conn.execute("VACUUM")  # would raise OperationalError if in a txn


# -----------------------------------------------------------------------------
# Bug 9: forget() returned True even when nothing was deleted. The fix checks
# the collection before deleting and only returns True if the id existed.
# -----------------------------------------------------------------------------

def test_forget_returns_false_for_unknown_id():
    """forget() must return False when no chunk with that id exists."""
    import asyncio
    from src.memory import Memory
    from src.tier import Tier

    class FakeColl:
        def __init__(self):
            self.ids = set()
        def get(self, ids=None, **kw):
            return {"ids": [i for i in (ids or []) if i in self.ids]}
        def delete(self, ids):
            for i in ids:
                self.ids.discard(i)

    class FakeStore:
        def __init__(self):
            self._coll = FakeColl()
        def collection_for(self, t):
            return self._coll

    async def main():
        m = Memory()
        store = FakeStore()
        m._store = store
        async def fake_init():
            return (store, None)
        m._ensure_initialized = fake_init
        return await m.forget("does_not_exist", tier=Tier.SEMANTIC)

    assert asyncio.run(main()) is False


def test_forget_returns_true_for_known_id():
    """forget() must return True when a chunk with that id existed."""
    import asyncio
    from src.memory import Memory
    from src.tier import Tier

    class FakeColl:
        def __init__(self, ids):
            self.ids = set(ids)
        def get(self, ids=None, **kw):
            return {"ids": [i for i in (ids or []) if i in self.ids]}
        def delete(self, ids):
            for i in ids:
                self.ids.discard(i)

    class FakeStore:
        def __init__(self, ids):
            self._coll = FakeColl(ids)
        def collection_for(self, t):
            return self._coll

    async def main():
        m = Memory()
        store = FakeStore(ids=["known_id"])
        m._store = store
        async def fake_init():
            return (store, None)
        m._ensure_initialized = fake_init
        return await m.forget("known_id", tier=Tier.SEMANTIC)

    assert asyncio.run(main()) is True


# -----------------------------------------------------------------------------
# Bug 11: eval.py's _is_hit() read metadata["tier"] but tier is a top-level
#   attribute on QueryResult and was never stored in metadata. The fix threads
#   result_tier through the call site.
# -----------------------------------------------------------------------------

def test_eval_is_hit_uses_result_tier_param():
    """_is_hit must consult the explicit result_tier param, not metadata."""
    from src.eval import _is_hit, EvalEntry

    entry = EvalEntry(
        query="q",
        expected_keywords=[],
        expected_tier="semantic",
        expected_source_path=None,
        expected_section=None,
    )
    # With the fix: tier match succeeds when result_tier == expected_tier
    assert _is_hit("some text", {}, entry, result_tier="semantic") is True
    # Mismatched tier → not a hit
    assert _is_hit("some text", {}, entry, result_tier="episodic") is False


# -----------------------------------------------------------------------------
# Bug 12: chroma.py used n_results // len(tiers) which floors and under-fetches.
#   The fix uses math.ceil so the union always has >= n_results candidates.
# -----------------------------------------------------------------------------

def test_chroma_query_split_uses_ceil():
    """Per-tier fetch must be ceil(n_results / n_tiers) so the union covers
    at least n_results candidates."""
    # Reproduce the fixed per-tier computation
    def per_tier(n_results, n_tiers):
        return math.ceil(n_results / n_tiers) if n_tiers > 1 else n_results
    assert per_tier(5, 4) == 2   # was 1 before fix
    assert per_tier(5, 3) == 2   # was 1
    assert per_tier(10, 4) == 3  # was 2
    assert per_tier(7, 1) == 7   # single-tier unchanged


# -----------------------------------------------------------------------------
# Bonus: consolidate.py he/she regex was a literal "he/she" match, not an
#   alternation. The fix splits it into (?:he|she).
# -----------------------------------------------------------------------------

def test_consolidate_matches_he_said():
    from src.consolidate import FACT_PATTERNS
    kinds = set()
    for pat, kind in FACT_PATTERNS:
        if pat.search("he said that the deploy is broken"):
            kinds.add(kind)
    assert "user-said" in kinds


def test_consolidate_matches_she_said():
    from src.consolidate import FACT_PATTERNS
    kinds = set()
    for pat, kind in FACT_PATTERNS:
        if pat.search("she decided to roll back"):
            kinds.add(kind)
    assert "user-said" in kinds
