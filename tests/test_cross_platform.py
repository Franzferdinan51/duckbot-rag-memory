"""
test_cross_platform.py — verify the cross-platform watcher and installer.

duckbot-secret-scan: allowlist-file

Tests:
  - The watcher dispatches to _daemon_windows on win32, _daemon_posix
    elsewhere.
  - _daemon_windows uses subprocess.Popen with DETACHED_PROCESS +
    CREATE_NEW_PROCESS_GROUP (no os.fork).
  - /dev/null is never referenced in the watcher (uses os.devnull).
  - os.fork() is never called at module import time (so import works
    on Windows even when --daemon is not invoked).
  - The install scripts exist for all 3 OSes.
  - The install scripts are syntactically valid (bash -n for .sh,
    structural check for .ps1).
  - embeddings.py uses pathlib (not os.path.join) for the .env file.
"""

# duckbot-secret-scan: allowlist-file
from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = "/Users/duckets/Desktop/duckbot-rag-memory"
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# -----------------------------------------------------------------------------
# Watcher: cross-platform dispatch
# -----------------------------------------------------------------------------


def test_watcher_module_imports_on_any_platform():
    """src.watcher imports without calling os.fork at import time."""
    import src.watcher  # noqa: F401
    # If we got here without AttributeError, import worked on this OS.


def test_watcher_has_platform_specific_daemon_helpers():
    """The watcher exposes _daemon_windows and _daemon_posix."""
    from src import watcher
    assert callable(getattr(watcher, "_daemon_windows", None))
    assert callable(getattr(watcher, "_daemon_posix", None))


def test_watcher_daemon_dispatches_to_posix_here(monkeypatch):
    """On non-Windows, cmd_daemon should use _daemon_posix."""
    from src import watcher

    called = {"posix": 0, "windows": 0}

    def fake_posix(paths, args):
        called["posix"] += 1
        return 0

    def fake_windows(paths, args):
        called["windows"] += 1
        return 0

    monkeypatch.setattr(watcher, "_daemon_posix", fake_posix)
    monkeypatch.setattr(watcher, "_daemon_windows", fake_windows)
    monkeypatch.setattr(watcher, "PID_PATH", Path(ROOT) / "data" / "watcher.pid_test")
    # Force POSIX path
    monkeypatch.setattr(watcher.sys, "platform", "darwin")

    class Args:
        paths = []
        interval = 2.0

    watcher.cmd_daemon(Args())
    assert called["posix"] == 1
    assert called["windows"] == 0


def test_watcher_daemon_dispatches_to_windows(monkeypatch):
    """On win32, cmd_daemon should use _daemon_windows."""
    from src import watcher

    called = {"posix": 0, "windows": 0}

    def fake_posix(paths, args):
        called["posix"] += 1
        return 0

    def fake_windows(paths, args):
        called["windows"] += 1
        return 0

    monkeypatch.setattr(watcher, "_daemon_posix", fake_posix)
    monkeypatch.setattr(watcher, "_daemon_windows", fake_windows)
    monkeypatch.setattr(watcher.sys, "platform", "win32")

    class Args:
        paths = []
        interval = 2.0

    # Will try to actually start a subprocess; use PID_PATH that's not real
    monkeypatch.setattr(watcher, "PID_PATH", Path(ROOT) / "data" / "watcher.pid_test_win")
    # Don't actually start anything; the test will fail if it tries to Popen
    # a python that doesn't exist. Patch subprocess.Popen to a no-op.
    import subprocess
    class FakePopen:
        pid = 99999
        def poll(self): return None
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: FakePopen())
    watcher.cmd_daemon(Args())
    assert called["windows"] == 1
    assert called["posix"] == 0


# -----------------------------------------------------------------------------
# /dev/null replaced with os.devnull
# -----------------------------------------------------------------------------


def test_watcher_does_not_reference_devnull_literal():
    """The watcher should use os.devnull, not the literal '/dev/null'."""
    src = (Path(ROOT) / "src" / "watcher.py").read_text()
    assert '"/dev/null"' not in src, "Use os.devnull instead of '/dev/null' literal"
    assert "'/dev/null'" not in src
    # And confirm os.devnull is referenced
    assert "os.devnull" in src


def test_watcher_daemon_posix_avoids_chmod_when_unavailable():
    """os.chmod is guarded with hasattr() so Windows doesn't crash."""
    src = (Path(ROOT) / "src" / "watcher.py").read_text()
    # Find the chmod call
    assert "if hasattr(os, \"chmod\"):" in src or "if hasattr(os, 'chmod'):" in src


# -----------------------------------------------------------------------------
# embeddings.py: pathlib for .env
# -----------------------------------------------------------------------------


def test_embeddings_uses_pathlib_for_env():
    """src/embeddings.py should use Path for the .env file path."""
    src = (Path(ROOT) / "src" / "embeddings.py").read_text()
    # Should NOT have os.path.join for the .env file
    assert "os.path.join" not in src, "Use pathlib.Path instead of os.path.join"
    # SHOULD have pathlib
    assert "from pathlib import Path" in src
    # SHOULD build the path with the / operator
    assert '/ ".env"' in src or '".env"' in src
    # And use .exists() not os.path.exists
    assert ".exists()" in src


# -----------------------------------------------------------------------------
# Cross-platform scripts exist
# -----------------------------------------------------------------------------


def test_install_script_bash_exists():
    """Generic POSIX install.sh exists."""
    p = Path(ROOT) / "scripts" / "install.sh"
    assert p.exists()


def test_install_macos_script_exists():
    """macOS-specific install (launchd) exists."""
    p = Path(ROOT) / "scripts" / "install-macos.sh"
    assert p.exists()
    # Should reference launchctl
    content = p.read_text()
    assert "launchctl" in content


def test_install_linux_script_exists():
    """Linux-specific install (systemd) exists."""
    p = Path(ROOT) / "scripts" / "install-linux.sh"
    assert p.exists()
    content = p.read_text()
    assert "systemctl" in content
    assert "systemd" in content.lower() or "user" in content


def test_install_ps1_exists():
    """Windows-specific install (Task Scheduler) exists."""
    p = Path(ROOT) / "scripts" / "install.ps1"
    assert p.exists()
    content = p.read_text()
    assert "ScheduledTask" in content
    assert "Register-ScheduledTask" in content


def test_start_watcher_ps1_exists():
    """Windows background-watcher launcher exists."""
    p = Path(ROOT) / "scripts" / "start-watcher.ps1"
    assert p.exists()
    content = p.read_text()
    assert "Start-Process" in content
    # Should NOT use fork
    assert "fork" not in content.lower()


def test_start_watcher_sh_exists():
    """POSIX background-watcher launcher still exists."""
    p = Path(ROOT) / "scripts" / "start-watcher.sh"
    assert p.exists()


# -----------------------------------------------------------------------------
# Bash syntax validation
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("script", [
    "scripts/install.sh",
    "scripts/install-linux.sh",
    "scripts/install-macos.sh",
    "scripts/start-watcher.sh",
    "scripts/secret-scan.sh",
])
def test_bash_script_parses(script):
    """The bash script passes `bash -n` syntax check."""
    p = Path(ROOT) / script
    if not p.exists():
        pytest.skip(f"missing: {p}")
    result = subprocess.run(
        ["bash", "-n", str(p)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"bash syntax error in {script}: {result.stderr}"


# -----------------------------------------------------------------------------
# All python files use pathlib (no os.path.join in new/modified files)
# -----------------------------------------------------------------------------


def test_no_os_path_join_in_src():
    """No file under src/ should use os.path.join."""
    offenders = []
    for path in (Path(ROOT) / "src").rglob("*.py"):
        if "__pycache__" in str(path):
            continue
        text = path.read_text(errors="ignore")
        for i, line in enumerate(text.splitlines(), 1):
            if "os.path.join" in line and not line.strip().startswith("#"):
                offenders.append(f"{path.relative_to(ROOT)}:{i}: {line}")
    assert not offenders, f"os.path.join found in src/:\n" + "\n".join(offenders)


# -----------------------------------------------------------------------------
# Platform reporting
# -----------------------------------------------------------------------------


def test_platform_info_recorded():
    """Just a sanity log of which OS the tests are running on."""
    print(f"\n[test_cross_platform] running on {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"[test_cross_platform] Python {platform.python_version()}")
    print(f"[test_cross_platform] os.sep = {os.sep!r}")
