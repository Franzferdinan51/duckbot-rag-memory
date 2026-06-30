#!/bin/bash
# Run the brain watcher under launchd.
#
# launchd quirks this script handles:
#   1. launchd gives a near-empty env (no PATH, no HOME). Set both before
#      doing anything else.
#   2. Without `.env`, the watcher can't auth to LM Studio and segfaults in
#      tokio the moment it tries to embed. Load `.env` from the repo root
#      BEFORE invoking python.
#   3. launchd captures exit codes, so `exec` straight to python and let
#      launchd track the real watcher PID.
#
# Note: do NOT use `set -u` or `set -e` — launchd may omit vars the script
# references, and we want ANY exit from the python process (even non-zero)
# to bubble up to launchd so KeepAlive can decide whether to relaunch.

# 1. Re-establish the basics (launchd gives us a barren env).
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export HOME="${HOME:-/Users/$(whoami)}"
export PYTHONUNBUFFERED=1

# 2. Resolve repo paths.
BRAIN_DIR="/Users/duckets/Desktop/duckbot-rag-memory"
OPENCLAW_DIR="${DUCKBOT_OPENCLAW_DIR:-$HOME/.openclaw/workspace}"

# Always use the absolute venv python path.
PYTHON="$BRAIN_DIR/.venv/bin/python"

# 3. Load `.env` from the repo root if it exists.
if [ -f "$BRAIN_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$BRAIN_DIR/.env"
    set +a
fi

# 4. Make sure data dir exists for the log file.
mkdir -p "$BRAIN_DIR/data" 2>/dev/null || true

# 5. Change into the brain repo so src.watcher can be found on the path.
cd "$BRAIN_DIR"

# 6. Exec the watcher directly. launchd inherits the python PID, so
#    KeepAlive actually watches the watcher. Output goes to data/watcher.log
#    so we can postmortem launchd failures after the fact.
exec "$PYTHON" -m src.watcher run \
    "$OPENCLAW_DIR/memory" \
    "$OPENCLAW_DIR/SOUL.md" \
    "$OPENCLAW_DIR/MEMORY.md" \
    "$OPENCLAW_DIR/USER.md" \
    "$OPENCLAW_DIR/AGENTS.md" \
    >> "$BRAIN_DIR/data/watcher.log" 2>&1
