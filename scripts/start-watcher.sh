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

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Default OPENCLAW paths if not set. The previous version only watched
# when the env var was already exported, which silently produced an
# empty watch list on fresh installs.
: "${OPENCLAW_MEMORY:=$HOME/.openclaw/workspace/memory}"
: "${OPENCLAW_WORKSPACE:=$HOME/.openclaw/workspace}"

# Load .env from the repo root if it exists. Pure bash parser so we don't
# need python+dotenv on the install path.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  . <(grep -vE '^\s*#' .env | grep -vE '^\s*$')
  set +a
fi

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

# Build watch list: prefer OPENCLAW_MEMORY from .env, fall back to the
# watcher's DEFAULT_WATCH (which is POSIX ~/.openclaw/workspace).
WATCH_ARGS=()
if [[ -n "${OPENCLAW_MEMORY:-}" && -e "${OPENCLAW_MEMORY}" ]]; then
  WATCH_ARGS+=("${OPENCLAW_MEMORY}")
fi
if [[ -n "${OPENCLAW_WORKSPACE:-}" && -e "${OPENCLAW_WORKSPACE}" ]]; then
  for f in AGENTS.md SOUL.md USER.md IDENTITY.md TOOLS.md MEMORY.md README.md CHANGELOG.md; do
    [[ -f "$OPENCLAW_WORKSPACE/$f" ]] && WATCH_ARGS+=("$OPENCLAW_WORKSPACE/$f")
  done
fi
# Always include the repo's own docs (they ship in the repo, so relative).
for f in AGENTS.md SOUL.md USER.md IDENTITY.md TOOLS.md README.md CHANGELOG.md; do
  [[ -f "$REPO_ROOT/$f" ]] && WATCH_ARGS+=("$REPO_ROOT/$f")
done

# nohup + redirect + & + disown: the classic "detach" recipe.
nohup "$PYTHON_BIN" -m src.watcher run "${WATCH_ARGS[@]}" \
  </dev/null >>"$REPO_ROOT/data/watcher.log" 2>&1 &
WPID=$!
disown $WPID 2>/dev/null || true
echo "Spawned pid=$WPID"
echo "Watching: ${WATCH_ARGS[*]:-<watcher defaults>}"
echo "Log: $REPO_ROOT/data/watcher.log"
echo "Status: $PYTHON_BIN -m src.watcher status"
echo "Stop:   $PYTHON_BIN -m src.watcher stop"
# Don't sleep here — the parent script will exit and the watcher must survive.
exit 0
