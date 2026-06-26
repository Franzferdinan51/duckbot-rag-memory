#!/usr/bin/env bash
# scripts/hermes-preflight.sh — manual brain_wake_up wrapper.
#
# Calls `brain_wake_up` on the duckbot-memory MCP server and prints a
# ready-to-paste context block to stdout.
#
# ⚠️  This script is NOT auto-invoked by Hermes. The MemoryProvider
# plugin (src/plugins/memory/duckbot_brain/) declares on_session_start
# + on_session_end hooks that fire automatically once it's activated
# via `memory.provider: duckbot-brain` in ~/.hermes/config.yaml.
#
# Use this script for manual / cron / one-shot invocations:
#
# Usage:
#   hermes-preflight.sh            # one-shot, prints context
#   hermes-preflight.sh --query X  # wake_up anchored on query X
#
# Requirements: same as the rest of the project (chromadb, src.cli, MCP
# server scripts). This script shells out to the MCP server via
# `python -m src.cli wake-up` — the CLI entry that v0.12.0 added.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Activate the venv (cross-platform; same pattern as duckbot-ask).
if [ -f "$REPO_ROOT/.venv/bin/python" ]; then
    PY="$REPO_ROOT/.venv/bin/python"
elif [ -f "$REPO_ROOT/.venv/Scripts/python.exe" ]; then
    PY="$REPO_ROOT/.venv/Scripts/python.exe"
else
    echo "❌ No venv python found at $REPO_ROOT/.venv/{bin,Scripts}/python" >&2
    exit 1
fi

# Forward optional --query
QUERY=""
if [[ "${1:-}" == "--query" && -n "${2:-}" ]]; then
    QUERY="$2"
fi

# Run the wake-up CLI. The CLI prints a formatted markdown block to
# stdout; we surface it as-is so it can be piped into the agent's
# context (e.g. via $(hermes-preflight.sh) in a Hermes skill manifest).
if [ -n "$QUERY" ]; then
    "$PY" -m src.cli wake-up --query "$QUERY"
else
    "$PY" -m src.cli wake-up
fi