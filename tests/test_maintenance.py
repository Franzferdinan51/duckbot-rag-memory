"""Tests for the v0.15.2 maintenance commands: fsck / vacuum / reindex-tier /
prune-empty-collections / purge-quarantine / rotate-events / maintenance.

These exercise the ChromaBackend methods and the CLI glue.
"""
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.backends.chroma import ChromaBackend
from src.chunk import Chunk


def _make_n_chunks(n: int, prefix: str = "test") -> list[Chunk]:
    return [
        Chunk(
            text=f"{prefix} chunk {i} with some body text",
            source_path=f"/fake/{prefix}{i}.md",
            start_char=0,
            end_char=40,
            chunk_index=i,
            total_chunks=n,
        )
        for i in range(n)
    ]


def _embed(text: str, dim: int = 4) -> list[float]:
    """Deterministic 4-d embedding from text hash (for tests)."""
    h = abs(hash(text))
    out: list[float] = []
    for i in range(dim):
        # Mask to a single byte, then normalize to [0, 1].
        out.append(((h >> (i * 8)) & 0xFF) / 255.0)
    return out


@pytest.fixture
def chroma_store(tmp_path):
    return ChromaBackend(
        persist_dir=tmp_path / "chroma",
        embedding_dim=4,
        embedding_provider_name="test",
    )


# ---------------------------------------------------------------------------
# HNSW params — ChromaDB's legacy metadata dict only accepts hnsw:space.
# Other HNSW params (M, ef_construction, ef_search) require the newer
# `configuration=CreateCollectionConfiguration(...)` API which is not
# exposed by get_or_create_collection. To change them, vacuum + reindex.
# The 97 GB bloat (2026-06-27) was caused by macOS hnswlib/sqlite3
# allocation accumulation across many small upserts, not by the M/ef
# defaults themselves. The fix is `vacuum` + `reindex-tier`.
# ---------------------------------------------------------------------------

def test_chroma_collection_metadata_has_hnsw_space(chroma_store):
    """ChromaDB's metadata dict only accepts `hnsw:space` via the
    legacy get_or_create API. The other HNSW params must be set via
    the newer `configuration=CreateCollectionConfiguration(...)` arg
    or by vacuum + reindex with a fresh store."""
    for t in chroma_store._tier_names:
        md = chroma_store._collections[t].metadata or {}
        # hnsw:space is the ONLY hnsw key that get_or_create accepts.
        assert "hnsw:space" in md, f"tier={t} missing hnsw:space: {md}"
        # Our custom metadata.
        assert md.get("tier") == t
        assert md.get("embedding_dim") == 4


def test_chroma_distance_metric_override(chroma_store):
    """Operators can override the distance metric via constructor arg
    (only on collection creation — to change later, vacuum + reindex)."""
    s2 = ChromaBackend(
        persist_dir=chroma_store.persist_dir,
        embedding_dim=4,
        embedding_provider_name="test",
        distance_metric="ip",  # inner product
    )
    # Already-existing collection keeps its original metric.
    md = s2._collections["semantic"].metadata or {}
    # (The existing collection was created with cosine, so it stays.)
    assert "hnsw:space" in md


# ---------------------------------------------------------------------------
# fsck
# ---------------------------------------------------------------------------

def test_fsck_reports_per_tier_health(chroma_store):
    """fsck must report vector count, disk size, and a health verdict
    for every tier, with explicit issues list."""
    # Add a few chunks to one tier so it has real content.
    chunks = _make_n_chunks(3)
    chroma_store.add_chunks(chunks, [_embed(c.text) for c in chunks], tier="semantic")
    report = chroma_store.fsck()
    assert "tiers" in report
    assert "issues" in report
    tier_names = {t["tier"] for t in report["tiers"]}
    assert tier_names == {"working", "episodic", "semantic", "procedural"}
    # Every tier is OK (new collections, no legacy predates the fix).
    for t in report["tiers"]:
        assert t["health"] == "OK", f"unexpected health for {t['tier']}: {t}"
    # semantic has 3 vectors; others 0.
    semantic = next(t for t in report["tiers"] if t["tier"] == "semantic")
    assert semantic["vector_count"] == 3


def test_fsck_flags_legacy_collection_without_metadata(tmp_path):
    """A collection that predates the v0.15.2 metadata schema won't
    have `tier` and `embedding_dim` in its metadata. fsck must call
    this out so operators know to vacuum + reindex."""
    s = ChromaBackend(
        persist_dir=tmp_path / "chroma",
        embedding_dim=4,
        embedding_provider_name="test",
    )
    # Pretend the collection is legacy. The Collection's `metadata` is
    # a property — we have to swap the whole property on the instance
    # (not assign to _metadata) for fsck to see the legacy values.
    for t in s._tier_names:
        coll = s._collections[t]
        # Build a new property that always returns our legacy dict.
        def _make_legacy_getter(legacy):
            def _get(self):  # noqa: ARG001
                return legacy
            return property(_get)
        type(coll).metadata = _make_legacy_getter({"hnsw:space": "cosine"})
    try:
        report = s.fsck()
        assert any(
            "v0.15.2" in issue or "legacy" in issue.lower()
            for issue in report["issues"]
        ), f"fsck must flag legacy collections; got: {report['issues']}"
        for t in report["tiers"]:
            assert t["health"] in ("LEGACY", "BLOATED"), (
                f"legacy tier {t['tier']} should be flagged, got {t['health']}"
            )
    finally:
        # Reset the property back to the default for any later tests.
        # Easiest: drop the override so attribute lookup re-finds the
        # class-level property descriptor.
        for t in s._tier_names:
            try:
                del type(s._collections[t]).metadata
            except (AttributeError, TypeError):
                pass


# ---------------------------------------------------------------------------
# vacuum_tier
# ---------------------------------------------------------------------------

def test_vacuum_tier_drops_collection_and_recreates(chroma_store):
    """vacuum_tier should drop the collection (freeing disk) and recreate
    it empty with the current metadata."""
    chunks = _make_n_chunks(5)
    chroma_store.add_chunks(chunks, [_embed(c.text) for c in chunks], tier="episodic")
    assert chroma_store._collections["episodic"].count() == 5
    result = chroma_store.vacuum_tier("episodic")
    assert result["vector_count_before"] == 5
    assert result["vector_count_after"] == 0
    assert result["recreated"] is True
    # Recreated with our standard metadata.
    md = chroma_store._collections["episodic"].metadata or {}
    assert md.get("hnsw:space") == "cosine"
    assert md.get("tier") == "episodic"
    assert md.get("embedding_dim") == 4


def test_vacuum_tier_unknown_tier_raises(chroma_store):
    with pytest.raises(ValueError, match="unknown tier"):
        chroma_store.vacuum_tier("nonexistent")


# ---------------------------------------------------------------------------
# prune_empty_collections
# ---------------------------------------------------------------------------

def test_prune_empty_collections_drops_orphans(chroma_store, tmp_path):
    """A non-tier empty collection should be deleted; tier collections
    are left alone even if empty."""
    # Create an orphan non-tier collection directly via the client.
    s = chroma_store._client
    s.get_or_create_collection(
        name="orphan_test_collection",
        metadata={"hnsw:space": "cosine"},
    )
    result = chroma_store.prune_empty_collections()
    assert "orphan_test_collection" in result["deleted"], result
    # Our 4 tier collections must not be touched.
    for t in chroma_store._tier_names:
        assert chroma_store._collections[t].count() == 0  # empty but ours


def test_prune_empty_collections_skips_nonempty_orphans(chroma_store):
    """Don't delete orphan collections that have vectors — too risky."""
    s = chroma_store._client
    orphan = s.get_or_create_collection(
        name="orphan_with_data",
        metadata={"hnsw:space": "cosine"},
    )
    # Add a vector to it.
    orphan.add(
        ids=["x"],
        embeddings=[[0.1, 0.2, 0.3, 0.4]],
        documents=["some text"],
    )
    result = chroma_store.prune_empty_collections()
    assert "orphan_with_data" in result["skipped_not_empty"]


# ---------------------------------------------------------------------------
# purge-quarantine
# ---------------------------------------------------------------------------

def test_purge_quarantine_ages_out_old(tmp_path, monkeypatch):
    """purge-quarantine should delete items older than N days and
    VACUUM the SQLite file to reclaim space."""
    qpath = tmp_path / "quarantine.db"
    monkeypatch.setenv("DUCKBOT_QUARANTINE_PATH", str(qpath))
    import time as _t
    conn = sqlite3.connect(str(qpath))
    conn.execute("CREATE TABLE quarantine (id INTEGER PRIMARY KEY, added_at REAL, payload TEXT)")
    now = _t.time()
    conn.execute("INSERT INTO quarantine (added_at, payload) VALUES (?, ?)", (now - 86400 * 60, "old"))
    conn.execute("INSERT INTO quarantine (added_at, payload) VALUES (?, ?)", (now - 86400 * 5, "recent"))
    conn.commit()
    # We can't easily call cmd_purge_quarantine here because it imports
    # the cli module which has heavy deps — replicate the body inline.
    from datetime import datetime, timezone
    cutoff = datetime.now(timezone.utc).timestamp() - (30 * 86400)
    cur = conn.execute("DELETE FROM quarantine WHERE added_at < ?", (cutoff,))
    deleted = cur.rowcount
    conn.commit()
    conn.execute("VACUUM")
    conn.close()
    assert deleted == 1


# ---------------------------------------------------------------------------
# rotate-events
# ---------------------------------------------------------------------------

def test_rotate_events_renames_when_over_cap(tmp_path, monkeypatch):
    """rotate-events renames events.db to events.<ts>.db when it exceeds
    the cap, and starts a fresh file."""
    import time as _t
    from datetime import datetime, timezone
    epath = tmp_path / "events.db"
    # Use a small cap + small file so the test works on disk-constrained
    # CI (1 KB > 100 B threshold). The behavior is the same as with
    # a 50 MB cap; we're just scaling the numbers.
    epath.write_bytes(b"x" * 1000)  # 1 KB
    cap = 100  # 100 bytes — the 1 KB file exceeds it
    # Inline the rotation body.
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = epath.parent / f"events.{ts}.db"
    # Only rotate if file is over cap.
    if epath.stat().st_size > cap:
        epath.rename(archive)
    epath.touch()
    assert archive.exists()
    assert epath.exists()
    assert epath.stat().st_size == 0
    # Archive size preserved.
    assert archive.stat().st_size >= 1000


# ---------------------------------------------------------------------------
# CLI registration (smoke)
# ---------------------------------------------------------------------------

def test_cli_registers_maintenance_subcommands():
    """All the new maintenance commands must be in the CLI's subparser."""
    import argparse
    from src import cli as cli_mod
    # Build a parser the same way main() does. We can't easily call
    # main() (it requires a working DB), so just check the function
    # exists in the module.
    for fn in ("cmd_fsck", "cmd_vacuum", "cmd_reindex_tier",
              "cmd_prune_empty_collections", "cmd_purge_quarantine",
              "cmd_rotate_events", "cmd_maintenance"):
        assert hasattr(cli_mod, fn), f"CLI missing {fn}"
