#!/bin/bash
# Run the brain watcher. Paths are templated:
#   ${DUCKBOT_BRAIN_DIR} defaults to the repo root (this script's parent dir).
#   ${DUCKBOT_OPENCLAW_DIR} defaults to ~/.openclaw/workspace.
#
# Set env vars to override; otherwise the script works out-of-the-box on
# this Mac and on any other machine.
#
# Launch behavior:
#   - If stdout is a TTY, run in the foreground (Ctrl-C stops it).
#   - If stdout is NOT a TTY (e.g. launchd, nohup), exec the python directly
#     so launchd's KeepAlive watches the actual watcher process. The
#     previous nohup + & pattern made the shell exit immediately,
#     leaving an orphan Python child that died on its own.
#
# launchd quirks this script handles:
#   1. launchd gives a near-empty env (no PATH, no HOME). Set both before
#      doing anything else.
#   2. Without `.env`, the watcher can't auth to LM Studio and segfaults in
#      tokio the moment it tries to embed. Load `.env` from the repo root
#      BEFORE invoking python.
#   3. launchd's stdout is NOT a TTY, so the "background" branch runs.
#
# Note: do NOT use `set -u` — launchd may omit vars the script references.

set -eo pipefail

# 1. Re-establish the basics (launchd gives us a barren env).
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export HOME="${HOME:-/Users/$(whoami)}"
export PYTHONUNBUFFERED=1

# 2. Resolve repo paths.
BRAIN_DIR="${DUCKBOT_BRAIN_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
OPENCLAW_DIR="${DUCKBOT_OPENCLAW_DIR:-$HOME/.openclaw/workspace}"

# Always use the absolute venv python path. Relative `.venv/bin/python`
# works when CWD is set, but breaks if anything in the parent ever changes
# cwd before the exec.
PYTHON="$BRAIN_DIR/.venv/bin/python"

# 3. Load `.env` from the repo root if it exists. Without this, LM Studio
# auth tokens, embedding-dim settings, and OpenClaw paths are missing
# under launchd, and the watcher crashes the moment it tries to embed
# anything (tokio segfaults on unauth'd httpx retries). Same pattern as
# start.sh — keep both launchers consistent.
if [ -f "$BRAIN_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1090,SC1091
    . "$BRAIN_DIR/.env"
    set +a
fi

cd "$BRAIN_DIR"

if [ -t 1 ]; then
    # Foreground: useful for ad-hoc testing.
    exec "$PYTHON" -m src.watcher run \
        "$OPENCLAW_DIR/memory" \
        "$OPENCLAW_DIR/SOUL.md" \
        "$OPENCLAW_DIR/MEMORY.md" \
        "$OPENCLAW_DIR/USER.md" \
        "$OPENCLAW_DIR/AGENTS.md"
else
    # Background (launchd, nohup, etc.): exec so launchd sees the real
    # watcher PID. Tee output to data/watcher.log for postmortem.
    mkdir -p "$BRAIN_DIR/data"
    exec "$PYTHON" -m src.watcher run \
        "$OPENCLAW_DIR/memory" \
        "$OPENCLAW_DIR/SOUL.md" \
        "$OPENCLAW_DIR/MEMORY.md" \
        "$OPENCLAW_DIR/USER.md" \
        "$OPENCLAW_DIR/AGENTS.md" \
        >> "$BRAIN_DIR/data/watcher.log" 2>&1
fi