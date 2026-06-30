#!/bin/bash
# Launchd-safe launcher for the brain watcher.
# The repo + .venv + .env live on ~/Desktop, which launchd blocks for
# writable file ops (gatekeeper/provenance). Logs go to a safe dir.
export PATH="/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
export HOME="${HOME:-/Users/duckets}"
export PYTHONUNBUFFERED=1

BRAIN_DIR="/Users/duckets/Desktop/duckbot-rag-memory"
PYTHON="$BRAIN_DIR/.venv/bin/python"
ENV_FILE="/Users/duckets/Library/Application Support/duckbot-rag-memory/env"
LOG_DIR="/Users/duckets/Library/Application Support/duckbot-rag-memory/logs"
LOG_FILE="$LOG_DIR/watcher.log"

# Source env from launchd-safe location.
if [ -f "$ENV_FILE" ]; then
    set -a
    . "$ENV_FILE"
    set +a
fi

mkdir -p "$LOG_DIR" 2>/dev/null || true
cd "$BRAIN_DIR"

exec "$PYTHON" -m src.watcher run \
    "$HOME/.openclaw/workspace/memory" \
    "$HOME/.openclaw/workspace/SOUL.md" \
    "$HOME/.openclaw/workspace/MEMORY.md" \
    "$HOME/.openclaw/workspace/USER.md" \
    "$HOME/.openclaw/workspace/AGENTS.md" \
    >> "$LOG_FILE" 2>&1
