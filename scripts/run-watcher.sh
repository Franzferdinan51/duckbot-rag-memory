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
#   - If stdout is NOT a TTY (e.g. launchd), exec the python directly so
#     launchd's KeepAlive watches the actual watcher process. The
#     previous nohup + & pattern made the shell exit immediately,
#     leaving an orphan Python child that died on its own.

set -euo pipefail

BRAIN_DIR="${DUCKBOT_BRAIN_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
OPENCLAW_DIR="${DUCKBOT_OPENCLAW_DIR:-$HOME/.openclaw/workspace}"

cd "$BRAIN_DIR"

if [ -t 1 ]; then
    # Foreground: useful for ad-hoc testing.
    exec .venv/bin/python -m src.watcher run \
        "$OPENCLAW_DIR/memory" \
        "$OPENCLAW_DIR/SOUL.md" \
        "$OPENCLAW_DIR/MEMORY.md" \
        "$OPENCLAW_DIR/USER.md" \
        "$OPENCLAW_DIR/AGENTS.md"
else
    # Background (launchd, nohup, etc.): exec so launchd sees the real
    # watcher PID. Tee output to data/watcher.log for postmortem.
    mkdir -p data
    exec .venv/bin/python -m src.watcher run \
        "$OPENCLAW_DIR/memory" \
        "$OPENCLAW_DIR/SOUL.md" \
        "$OPENCLAW_DIR/MEMORY.md" \
        "$OPENCLAW_DIR/USER.md" \
        "$OPENCLAW_DIR/AGENTS.md" \
        >> data/watcher.log 2>&1
fi