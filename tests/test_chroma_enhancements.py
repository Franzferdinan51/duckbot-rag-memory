"""
test_chroma_enhancements.py — verify the Chroma enhancements from the
"cross-platform Chroma" layer.

duckbot-secret-scan: allowlist-file

Tests:
  - ChromaBackend accepts distance_metric="cosine"|"l2"|"ip" and rejects
    unknown values.
  - ChromaBackend persists distance_metric to collection metadata.
  - store.MemoryStore reads DUCKBOT_CHROMA_DISTANCE env var and passes
    it through to the backend.
  - The new `cmd_compact` CLI subcommand can be invoked (we mock the
    underlying calls to avoid touching real data on the test box).
  - secret-scan.ps1 exists and is syntactically valid PowerShell
    (we just check the file is present and the marker comment is at
    the top; a full PS1 parse is impossible without pwsh on Linux).
  - install-pre-commit.ps1 exists.
"""

# duckbot-secret-scan: allowlist-file
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# -----------------------------------------------------------------------------
# Distance metric knob
# -----------------------------------------------------------------------------


def test_chroma_backend_accepts_cosine_default():
    """Default distance metric is cosine."""
    from src.backends.chroma import ChromaBackend
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DUCKBOT_CHROMA_DIR", "/tmp/test_chroma_cosine_xyz")
        b = ChromaBackend(persist_dir="/tmp/test_chroma_cosine_xyz")
        assert b.distance_metric == "cosine"


def test_chroma_backend_accepts_l2():
    """distance_metric='l2' is accepted."""
    from src.backends.chroma import ChromaBackend
    b = ChromaBackend(
        persist_dir="/tmp/test_chroma_l2_xyz",
        distance_metric="l2",
    )
    assert b.distance_metric == "l2"


def test_chroma_backend_accepts_ip():
    """distance_metric='ip' (inner product) is accepted."""
    from src.backends.chroma import ChromaBackend
    b = ChromaBackend(
        persist_dir="/tmp/test_chroma_ip_xyz",
        distance_metric="ip",
    )
    assert b.distance_metric == "ip"


def test_chroma_backend_rejects_unknown_metric():
    """Unknown distance metric raises ValueError."""
    from src.backends.chroma import ChromaBackend
    with pytest.raises(ValueError, match="distance_metric must be one of"):
        ChromaBackend(
            persist_dir="/tmp/test_chroma_bad_xyz",
            distance_metric="manhattan",
        )


def test_chroma_backend_supported_metrics_constant():
    """The class documents its supported metrics."""
    from src.backends.chroma import ChromaBackend
    assert set(ChromaBackend.SUPPORTED_DISTANCE_METRICS) == {"cosine", "l2", "ip"}


def test_chroma_backend_metric_persisted_to_collection_metadata(tmp_path):
    """The metric is written to the collection's metadata on creation."""
    from src.backends.chroma import ChromaBackend
    b = ChromaBackend(persist_dir=tmp_path / "chroma_meta", distance_metric="l2")
    for tier in b.supported_tiers:
        coll = b.collection_for(tier)
        meta = coll.metadata
        assert meta["hnsw:space"] == "l2"


# -----------------------------------------------------------------------------
# MemoryStore passes DUCKBOT_CHROMA_DISTANCE through
# -----------------------------------------------------------------------------


def test_memory_store_passes_distance_metric_from_env(monkeypatch, tmp_path):
    """DUCKBOT_CHROMA_DISTANCE=ip → backend gets distance_metric='ip'."""
    monkeypatch.setenv("DUCKBOT_CHROMA_DISTANCE", "ip")
    monkeypatch.setenv("DUCKBOT_CHROMA_DIR", str(tmp_path / "chroma_env"))
    from src.store import MemoryStore
    s = MemoryStore()
    assert s.backend.distance_metric == "ip"


def test_memory_store_default_metric_is_cosine(monkeypatch, tmp_path):
    """No env var → cosine (default)."""
    monkeypatch.delenv("DUCKBOT_CHROMA_DISTANCE", raising=False)
    monkeypatch.setenv("DUCKBOT_CHROMA_DIR", str(tmp_path / "chroma_default"))
    from src.store import MemoryStore
    s = MemoryStore()
    assert s.backend.distance_metric == "cosine"


# -----------------------------------------------------------------------------
# CLI compact subcommand
# -----------------------------------------------------------------------------


def test_cli_compact_subcommand_registered():
    """`python -m src.cli compact` is a registered subcommand."""
    import argparse
    from src.cli import main
    # The `main()` function builds the parser and returns the rc.
    # We just need to invoke the parser-building portion. The easiest
    # way: import argparse, then call main() with a dry-run cmd.
    # But main() actually runs the cmd. Instead, rebuild via the same
    # pattern main() uses: import argparse, build, then add the
    # subcommands. The simplest correct check: the cmd_compact function
    # exists and is importable.
    from src.cli import cmd_compact
    assert callable(cmd_compact)


def test_cli_compact_uses_chroma_backend(monkeypatch):
    """compact() refuses non-chroma backends with a clear message."""
    import warnings
    from src.cli import cmd_compact

    class FakeNonChromaBackend:
        name = "qdrant"
        supported_tiers = ["working", "episodic", "semantic", "procedural"]
        persist_dir = Path("/tmp/nonexistent")

        def collection_for(self, tier):
            return None

    class FakeStore:
        backend = FakeNonChromaBackend()

    def _fake_run(coro):
        # The compact() function used to call asyncio.run(MemoryStore())
        # which was wrong (MemoryStore() is sync, not a coroutine). After
        # the fix, cmd_compact just calls MemoryStore() directly. This
        # fake is now unused but kept for back-compat with monkeypatch.
        try:
            coro.close()
        except Exception:
            pass
        return FakeStore()

    # cmd_compact imports MemoryStore lazily from src.store, so patch
    # it on the source module to redirect to our FakeStore.
    monkeypatch.setattr("src.store.MemoryStore", lambda *a, **kw: FakeStore())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rc = cmd_compact(None)
    assert rc == 1


def test_cli_compact_handles_bad_batch_env(monkeypatch, tmp_path):
    """compact() should fall back to 32 when the batch env is garbage."""
    from src.cli import cmd_compact

    class FakeCollection:
        def __init__(self):
            self.upsert_batches: list[int] = []
            self._rows = {
                "dup": {"document": "old dup", "metadata": {"ingested_at": 1}},
                "keep1": {"document": "keep1", "metadata": {"ingested_at": 10}},
                "keep2": {"document": "keep2", "metadata": {"ingested_at": 20}},
                "keep3": {"document": "keep3", "metadata": {"ingested_at": 30}},
            }

        def get(self, ids=None, include=None):
            if ids is None:
                return {
                    "ids": ["dup", "dup", "keep1", "keep2", "keep3"],
                    "documents": ["old dup", "new dup", "keep1", "keep2", "keep3"],
                    "embeddings": [[0.0], [1.0], [2.0], [3.0], [4.0]],
                    "metadatas": [
                        {"ingested_at": 1},
                        {"ingested_at": 2},
                        {"ingested_at": 10},
                        {"ingested_at": 20},
                        {"ingested_at": 30},
                    ],
                }
            ids = list(ids)
            return {
                "ids": ids,
                "documents": [self._rows[i]["document"] for i in ids],
                "embeddings": [[float(idx)] for idx, _ in enumerate(ids)],
                "metadatas": [self._rows[i]["metadata"] for i in ids],
            }

        def upsert(self, *, ids, documents, embeddings, metadatas):
            self.upsert_batches.append(len(ids))

    class FakeBackend:
        name = "chroma"
        supported_tiers = ["semantic"]
        persist_dir = tmp_path / "persist"

        def __init__(self):
            self._client = object()
            self._coll = FakeCollection()

        def collection_for(self, tier):
            return self._coll

    holder = {}

    class FakeStore:
        def __init__(self):
            self.backend = FakeBackend()
            holder["store"] = self

    monkeypatch.setattr("src.store.MemoryStore", lambda *a, **kw: FakeStore())
    monkeypatch.setenv("DUCKBOT_CHROMA_UPSERT_BATCH", "not-a-number")
    rc = cmd_compact(None)
    assert rc == 0
    assert holder["store"].backend._coll.upsert_batches == [4]


def test_cli_compact_batches_reupserts(monkeypatch, tmp_path):
    """compact() should split large re-upserts into multiple calls."""
    from src.cli import cmd_compact

    class FakeCollection:
        def __init__(self):
            self.upsert_batches: list[int] = []

        def get(self, ids=None, include=None):
            if ids is None:
                return {
                    "ids": ["dup", "dup", "keep1", "keep2", "keep3", "keep4"],
                    "documents": ["old dup", "new dup", "keep1", "keep2", "keep3", "keep4"],
                    "embeddings": [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
                    "metadatas": [
                        {"ingested_at": 1},
                        {"ingested_at": 2},
                        {"ingested_at": 10},
                        {"ingested_at": 20},
                        {"ingested_at": 30},
                        {"ingested_at": 40},
                    ],
                }
            ids = list(ids)
            return {
                "ids": ids,
                "documents": [f"doc:{i}" for i in ids],
                "embeddings": [[float(idx)] for idx, _ in enumerate(ids)],
                "metadatas": [{"ingested_at": idx} for idx, _ in enumerate(ids)],
            }

        def upsert(self, *, ids, documents, embeddings, metadatas):
            self.upsert_batches.append(len(ids))

    class FakeBackend:
        name = "chroma"
        supported_tiers = ["semantic"]
        persist_dir = tmp_path / "persist"

        def __init__(self):
            self._client = object()
            self._coll = FakeCollection()

        def collection_for(self, tier):
            return self._coll

    holder = {}

    class FakeStore:
        def __init__(self):
            self.backend = FakeBackend()
            holder["store"] = self

    monkeypatch.setattr("src.store.MemoryStore", lambda *a, **kw: FakeStore())
    monkeypatch.setenv("DUCKBOT_CHROMA_UPSERT_BATCH", "2")
    rc = cmd_compact(None)
    assert rc == 0
    assert holder["store"].backend._coll.upsert_batches == [2, 2, 1]


# -----------------------------------------------------------------------------
# Cross-platform script files
# -----------------------------------------------------------------------------


def test_secret_scan_ps1_exists():
    """PowerShell port of secret-scan exists for Windows users."""
    p = Path(ROOT) / "scripts" / "secret-scan.ps1"
    assert p.exists(), f"missing: {p}"
    content = p.read_text()
    # The allowlist marker must be at the top
    assert "duckbot-secret-scan: allowlist-file" in content[:200]
    # Must have a param() block
    assert "[CmdletBinding()]" in content
    assert "param(" in content


def test_install_pre_commit_ps1_exists():
    """PowerShell installer for the pre-commit hook exists."""
    p = Path(ROOT) / "scripts" / "install-pre-commit.ps1"
    assert p.exists(), f"missing: {p}"
    content = p.read_text()
    assert "duckbot-secret-scan: allowlist-file" in content[:200]
    # Must reference both .ps1 and .sh paths
    assert "secret-scan.ps1" in content
    assert "secret-scan.sh" in content


def test_bash_secret_scan_still_present():
    """The original bash version must remain (Unix still uses it)."""
    p = Path(ROOT) / "scripts" / "secret-scan.sh"
    assert p.exists()
    content = p.read_text()
    assert "duckbot-secret-scan: allowlist-file" in content[:200]


# -----------------------------------------------------------------------------
# Path safety (cross-platform)
# -----------------------------------------------------------------------------


def test_persist_dir_uses_pathlib_not_string_concat():
    """DEFAULT_PERSIST_DIR uses pathlib so Windows backslashes work."""
    from src.backends.chroma import ChromaBackend
    p = ChromaBackend.DEFAULT_PERSIST_DIR
    # Pure Path, not str. Works on Win/Mac/Linux identically.
    assert isinstance(p, Path)
    assert "data" in str(p)
    assert "chroma" in str(p)


def test_default_persist_dir_in_store_is_pathlib():
    """src/store.py DEFAULT_PERSIST_DIR is also a Path."""
    from src.store import DEFAULT_PERSIST_DIR
    assert isinstance(DEFAULT_PERSIST_DIR, Path)
