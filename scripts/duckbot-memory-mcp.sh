#!/usr/bin/env bash
# Hermes MCP launcher for duckbot-brain.
# Loads the brain's .env and starts the stdio MCP server.
# Cross-platform: works on macOS/Linux (.venv/bin/python) and
# Windows git-bash/PowerShell (.venv/Scripts/python.exe).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Load .env if present (LMSTUDIO_KEY, OPENAI_API_KEY, MINIMAX_API_KEY, etc.)
if [ -f "$REPO_ROOT/.env" ]; then
    # shellcheck disable=SC1091
    set -a
    . "$REPO_ROOT/.env"
    set +a
fi

# Ensure PYTHONUNBUFFERED so stdio flushes promptly (Hermes reads line-by-line).
export PYTHONUNBUFFERED=1

# Detect venv python path — Windows venv uses Scripts/, POSIX uses bin/.
if [ -x "$REPO_ROOT/.venv/Scripts/python.exe" ]; then
    PYTHON_BIN="$REPO_ROOT/.venv/Scripts/python.exe"
elif [ -x "$REPO_ROOT/.venv/Scripts/python" ]; then
    PYTHON_BIN="$REPO_ROOT/.venv/Scripts/python"
elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
    echo "❌ No venv python found at $REPO_ROOT/.venv/{bin,Scripts}/python" >&2
    exit 1
fi

exec "$PYTHON_BIN" -m src.mcp_server "$@"