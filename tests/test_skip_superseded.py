"""Regression test for the 'looping' bug in brain_recall.

2026-06-30 14:43 EDT — Duckets caught brain_inflate returning the same text
4-5 times because brain_recall wasn't filtering out chunks marked with
`superseded_by` metadata. The wake_up() helper had a filter but the
underlying recall() and the MCP/connector entry points didn't.

This test calls Memory.recall() directly (the Python API) to verify the
filter logic. The MCP-level integration is covered by the running duckbot-memory
MCP server, which gets the new code on next process restart.
"""
import sys
import time
import asyncio
import pytest

sys.path.insert(0, "/Users/duckets/Desktop/duckbot-rag-memory")


@pytest.mark.asyncio
async def test_recall_drops_superseded_by_default():
    """The core 'looping fix': recall should not return superseded chunks."""
    from src.memory import Memory

    m = Memory()

    query = f"looping-fix-test-{int(time.time())}"

    # Seed two chunks with similar content
    fact1 = (
        f"# TestLoopingFixA\n\n"
        f"Looping test version A for query '{query}'. "
        f"This text describes the first version of a fact about looping fix tests."
    )
    fact2 = (
        f"# TestLoopingFixB\n\n"
        f"Looping test version B for query '{query}'. "
        f"This text describes the second (updated) version of a fact about looping fix tests."
    )

    r1 = await m.remember(fact1, source_path="looping_fix_test.md")
    r2 = await m.remember(fact2, source_path="looping_fix_test.md")
    assert r1.chunk_id
    assert r2.chunk_id
    assert r1.chunk_id != r2.chunk_id
    id1, id2 = r1.chunk_id, r2.chunk_id

    # Manually mark id1 as superseded (brain_supersede requires tier routing
    # that we don't want to depend on for the unit test)
    store = m._store
    from src.tier import Tier
    # Both facts likely landed in semantic (via auto-tier from the "# Title" header)
    coll = store.collection_for(Tier.SEMANTIC)
    cur = coll.get(ids=[id1], include=["metadatas"])
    if cur and cur["ids"]:
        md = dict(cur["metadatas"][0])
        md["superseded_by"] = id2
        md["superseded_at"] = time.time()
        coll.update(ids=[id1], metadatas=[md])
    else:
        # Maybe landed in episodic — try that too
        coll = store.collection_for(Tier.EPISODIC)
        cur = coll.get(ids=[id1], include=["metadatas"])
        if cur and cur["ids"]:
            md = dict(cur["metadatas"][0])
            md["superseded_by"] = id2
            md["superseded_at"] = time.time()
            coll.update(ids=[id1], metadatas=[md])

    try:
        # Default recall (skip_superseded=True) should NOT return id1
        results, stats = await m.recall(query, k=10)
        result_ids = [r.chunk_id for r in results]
        assert id1 not in result_ids, (
            f"Looping bug! Superseded chunk {id1} should NOT be in recall results: {result_ids}"
        )
        # Stats should report how many were filtered
        assert stats.superseded_filtered >= 1, (
            f"stats should report superseded_filtered >= 1, got {stats.superseded_filtered}"
        )

        # skip_superseded=False should return BOTH
        results_all, _ = await m.recall(query, k=10, skip_superseded=False)
        result_ids_all = [r.chunk_id for r in results_all]
        assert id1 in result_ids_all, (
            f"With skip_superseded=False, superseded chunk {id1} SHOULD be present"
        )
        assert id2 in result_ids_all, (
            f"With skip_superseded=False, replacement chunk {id2} should also be present"
        )
    finally:
        # Cleanup
        for cid in (id1, id2):
            try:
                await m.forget(cid)
            except Exception:
                pass


@pytest.mark.asyncio
async def test_query_stats_includes_superseded_filtered():
    """QueryStats should expose superseded_filtered counter for observability."""
    from src.memory import Memory

    m = Memory()
    # Use a very unlikely query so the count should be near 0
    results, stats = await m.recall("zzzzzzzzz-unlikely-query-12345", k=5)
    assert hasattr(stats, "superseded_filtered"), (
        f"QueryStats should have superseded_filtered attr, has: {dir(stats)}"
    )
    # The to_dict should also expose it
    d = stats.to_dict()
    assert "superseded_filtered" in d, f"to_dict missing superseded_filtered: {d}"
    # Value should be a non-negative int
    assert isinstance(d["superseded_filtered"], int)
    assert d["superseded_filtered"] >= 0
