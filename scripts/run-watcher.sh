#!/bin/bash
# Run the brain watcher as a background daemon. Paths are templated:
#   ${DUCKBOT_BRAIN_DIR} defaults to the repo root (this script's parent dir).
#   ${DUCKBOT_OPENCLAW_DIR} defaults to ~/.openclaw/workspace.
#
# Set env vars to override; otherwise the script works out-of-the-box on
# this Mac and on any other machine.

set -euo pipefail

BRAIN_DIR="${DUCKBOT_BRAIN_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
OPENCLAW_DIR="${DUCKBOT_OPENCLAW_DIR:-$HOME/.openclaw/workspace}"

cd "$BRAIN_DIR"

nohup .venv/bin/python -m src.watcher run \
    "$OPENCLAW_DIR/memory" \
    "$OPENCLAW_DIR/SOUL.md" \
    "$OPENCLAW_DIR/MEMORY.md" \
    "$OPENCLAW_DIR/USER.md" \
    "$OPENCLAW_DIR/AGENTS.md" \
    >> data/watcher.log 2>&1 &

echo "Watcher started in background (pid=$!, log=$BRAIN_DIR/data/watcher.log)"