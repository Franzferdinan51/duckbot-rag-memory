"""Regression tests for the `duck-memory` shell launcher.

Bug: 2026-06-27, invoking `./duck-memory` from any cwd other than the
repo root failed with `ModuleNotFoundError: No module named 'src'`
because Python didn't know to look in REPO_ROOT for the package.
Fix: set PYTHONPATH=REPO_ROOT in the launcher.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DUCK_MEMORY = REPO_ROOT / "duck-memory"


def test_duck_memory_launcher_sets_pythonpath():
    """The shell launcher must export PYTHONPATH pointing at the repo
    root so `import src.cli` works regardless of the caller's cwd."""
    text = DUCK_MEMORY.read_text()
    assert "PYTHONPATH" in text, (
        "duck-memory must set PYTHONPATH so 'import src.cli' works from "
        "any cwd. Without it, running ./duck-memory from ~/ or any "
        "other directory fails with ModuleNotFoundError."
    )
    # The PYTHONPATH should reference the repo root, not be hardcoded.
    assert "REPO_ROOT" in text and "PYTHONPATH" in text
    # The Python invocation must come AFTER the PYTHONPATH export.
    py_idx = text.find('"$PYTHON" -m src.cli')
    pp_idx = text.find("PYTHONPATH=")
    assert 0 <= pp_idx < py_idx, (
        f"PYTHONPATH export (idx {pp_idx}) must precede Python call "
        f"(idx {py_idx}). Without this, Python doesn't see the new "
        "PYTHONPATH and import fails."
    )


def test_duck_memory_bat_sets_pythonpath():
    """Same fix for the Windows .bat launcher."""
    bat = REPO_ROOT / "duck-memory.bat"
    if not bat.exists():
        pytest.skip("duck-memory.bat not present")
    text = bat.read_text()
    assert "PYTHONPATH" in text, (
        "duck-memory.bat must also set PYTHONPATH for cross-platform parity"
    )


def test_duck_memory_runs_from_different_cwd(tmp_path):
    """End-to-end: invoking `./duck-memory --help` from a non-repo
    directory must still find src.cli. Regression for the 2026-06-27
    ModuleNotFoundError. We use `--help` because it doesn't require
    any external services (LM Studio, git fetch, etc.) and completes
    instantly."""
    # Set env to disable the background _update_check git fetch (which
    # hangs under network/disk pressure and would mask the import test).
    env = {
        **os.environ,
        "PATH": str(REPO_ROOT / ".venv/bin") + ":" + os.environ.get("PATH", ""),
    }
    result = subprocess.run(
        [str(DUCK_MEMORY), "--help"],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),  # NOT the repo root
        timeout=15,
        env=env,
    )
    # We don't care if the help text is perfect — we only care that
    # the Python import didn't fail with the ModuleNotFoundError.
    assert "No module named 'src'" not in result.stderr, (
        f"duck-memory failed to import src.cli when invoked from a "
        f"non-repo cwd. stderr:\n{result.stderr[:500]}"
    )
    assert "ModuleNotFoundError" not in result.stderr
    # And the CLI did start — the help text mentions some subcommand.
    assert "usage:" in result.stdout.lower()
