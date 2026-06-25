#!/usr/bin/env bash
# scripts/hermes-postflight.sh — Hermes agent session-end hook.
#
# Triggers brain_reflect so anything the agent learned this session
# gets consolidated into the semantic tier. Designed to be invoked
# from ~/.hermesrc or as a SessionEnd hook.
#
# Usage:
#   hermes-postflight.sh            # default reflect (lookback 7 days)
#   hermes-postflight.sh --days 3   # reflect over last 3 days

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ -f "$REPO_ROOT/.venv/bin/python" ]; then
    PY="$REPO_ROOT/.venv/bin/python"
elif [ -f "$REPO_ROOT/.venv/Scripts/python.exe" ]; then
    PY="$REPO_ROOT/.venv/Scripts/python.exe"
else
    echo "❌ No venv python found at $REPO_ROOT/.venv/{bin,Scripts}/python" >&2
    exit 1
fi

DAYS=7
if [[ "${1:-}" == "--days" && -n "${2:-}" ]]; then
    DAYS="$2"
fi

# Run the reflect CLI.
"$PY" -m src.cli reflect --days "$DAYS" || {
    echo "⚠ reflect failed (non-fatal)" >&2
    exit 0
}