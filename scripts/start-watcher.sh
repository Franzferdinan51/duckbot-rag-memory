#!/usr/bin/env bash
# start-watcher.sh — start the DuckBot memory watcher fully detached from
# the calling shell.
#
# duckbot-secret-scan: allowlist-file
#
# Cross-platform companion: scripts/start-watcher.ps1 (Windows).
#
# This is the "nohup + disown" launcher. For auto-restart on crash/boot,
# use scripts/install-macos.sh (launchd plist) or scripts/install-linux.sh
# (systemd user unit) instead.
#
# Usage (from repo root):
#   ./scripts/start-watcher.sh
#
# State files (in data/):
#   data/watcher.pid  — PID written by the watcher itself in cmd_run
#   data/watcher.log  — append-only log

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Detect the venv python
if [[ -x .venv/bin/python ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "❌ No python found. Run scripts/install.sh first." >&2
  exit 1
fi

# Remove a stale pid file (the watcher will write a fresh one).
rm -f data/watcher.pid
mkdir -p data

# nohup + redirect + & + disown: the classic "detach" recipe.
nohup "$PYTHON_BIN" -m src.watcher run </dev/null >>"$REPO_ROOT/data/watcher.log" 2>&1 &
WPID=$!
disown $WPID 2>/dev/null || true
echo "Spawned pid=$WPID"
echo "Log: $REPO_ROOT/data/watcher.log"
echo "Status: $PYTHON_BIN -m src.watcher status"
echo "Stop:   $PYTHON_BIN -m src.watcher stop"
# Don't sleep here — the parent script will exit and the watcher must survive.
exit 0
