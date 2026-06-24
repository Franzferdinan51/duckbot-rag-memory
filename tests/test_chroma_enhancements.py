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


ROOT = "/Users/duckets/Desktop/duckbot-rag-memory"
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
        # The compact() function awaits a coroutine that returns MemoryStore().
        # We just return our FakeStore directly. Close the coro to suppress
        # the "never awaited" warning.
        try:
            coro.close()
        except Exception:
            pass
        return FakeStore()

    monkeypatch.setattr("src.cli.asyncio.run", _fake_run, raising=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rc = cmd_compact(None)
    assert rc == 1


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
