"""Regression test for the 'looping' bug in brain_recall.

2026-06-30 14:43 EDT — Duckets caught brain_inflate returning the same text
4-5 times because brain_recall wasn't filtering out chunks marked with
`superseded_by` metadata. The wake_up() helper had a filter but the
underlying recall() and the MCP/connector entry points didn't.

This test seeds two chunks with identical text, supersedes the first,
and verifies that recall() drops the superseded one by default but
returns both when skip_superseded=False.
"""
import sys
import time
import pytest

sys.path.insert(0, "/Users/duckets/Desktop/duckbot-rag-memory")


def test_recall_drops_superseded_by_default():
    """The core 'looping fix': recall should not return superseded chunks."""
    from src.connectors.openclaw import handle
    from src.connectors.base import Brain

    # Use a unique query for this test
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

    r1 = handle("brain_remember", {"text": fact1, "source_path": "looping_fix_test.md"})
    r2 = handle("brain_remember", {"text": fact2, "source_path": "looping_fix_test.md"})
    assert r1.get("stored"), f"first remember failed: {r1}"
    assert r2.get("stored"), f"second remember failed: {r2}"
    id1 = r1["chunk_id"]
    id2 = r2["chunk_id"]
    assert id1 != id2

    # Supersede the first with the second
    supersede_result = handle("brain_supersede", {"old_chunk_id": id1, "new_chunk_id": id2})
    # brain_supersede may not exist; if so, use manual approach
    if "error" in supersede_result and "unknown" in str(supersede_result.get("error", "")).lower():
        # Manual supersede via direct metadata write
        import chromadb
        from pathlib import Path
        from src.store import MemoryStore
        from src.tier import Tier
        store = MemoryStore()
        coll = store.collection_for(Tier.WORKING)  # force-tier semantic
        cur = coll.get(ids=[id1], include=["metadatas"])
        if cur and cur["ids"]:
            md = dict(cur["metadatas"][0])
            md["superseded_by"] = id2
            md["superseded_at"] = time.time()
            coll.update(ids=[id1], metadatas=[md])
            print(f"Manually superseded {id1} -> {id2}")

    try:
        # Default recall (skip_superseded=True) should NOT return id1
        r = handle("brain_recall", {"query": query, "k": 10})
        assert isinstance(r, dict) and "results" in r, f"recall returned: {r}"
        result_ids = [x.get("id") for x in r["results"]]
        assert id1 not in result_ids, (
            f"Looping bug! Superseded chunk {id1} should NOT be in recall results: {result_ids}"
        )
        # Stats should report how many were filtered
        stats = r.get("stats", {})
        filtered = stats.get("superseded_filtered", 0)
        assert filtered >= 1, f"stats should report superseded_filtered >= 1, got {filtered}"

        # skip_superseded=False should return BOTH
        r_all = handle("brain_recall", {"query": query, "k": 10, "skip_superseded": False})
        result_ids_all = [x.get("id") for x in r_all["results"]]
        assert id1 in result_ids_all, (
            f"With skip_superseded=False, superseded chunk {id1} SHOULD be present: {result_ids_all}"
        )
        assert id2 in result_ids_all, (
            f"With skip_superseded=False, replacement chunk {id2} should also be present"
        )
    finally:
        # Cleanup
        for cid in (id1, id2):
            try:
                handle("brain_forget", {"chunk_id": cid})
            except Exception:
                pass
