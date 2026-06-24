#!/usr/bin/env bash
# Hermes MCP launcher for duckbot-brain.
# Loads the brain's .env and starts the stdio MCP server.
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

exec "$REPO_ROOT/.venv/bin/python" -m src.mcp_server "$@"