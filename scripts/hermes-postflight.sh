#!/usr/bin/env bash
# scripts/hermes-postflight.sh — manual brain_reflect wrapper.
#
# Triggers brain_reflect so anything the agent learned this session
# gets consolidated into the semantic tier.
#
# ⚠️  This script is NOT auto-invoked by Hermes. The MemoryProvider
# plugin (src/plugins/memory/duckbot_brain/) declares on_session_start
# + on_session_end hooks that fire automatically once it's activated
# via `memory.provider: duckbot-brain` in ~/.hermes/config.yaml. The
# plugin's on_session_end handles durable-rule extraction already;
# use THIS script for cron-driven deep consolidation (reflect() over
# the last N days).
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