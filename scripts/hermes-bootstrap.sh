#!/usr/bin/env bash
# scripts/hermes-bootstrap.sh — one-command Hermes Agent → brain setup.
#
# Mirrors openclaw-bootstrap.sh but for Hermes Agent. Ingest every
# markdown file in Hermes's workspace (~/.hermes/memories/ by default)
# into the brain, then register the brain as an MCP server. The
# pre-flight + post-flight hooks (hermes-preflight.sh, hermes-postflight.sh)
# wire brain_wake_up to Hermes session start so the agent loads the
# full corpus on every session.
#
# Usage:
#   ./scripts/hermes-bootstrap.sh                     # default workspace
#   HERMES_HOME=/custom/hermes ./scripts/hermes-bootstrap.sh  # env override

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Activate the venv (cross-platform).
if [ -f "$REPO_ROOT/.venv/bin/python" ]; then
    PY="$REPO_ROOT/.venv/bin/python"
elif [ -f "$REPO_ROOT/.venv/Scripts/python.exe" ]; then
    PY="$REPO_ROOT/.venv/Scripts/python.exe"
else
    echo "❌ No venv found at $REPO_ROOT/.venv/{bin,Scripts}/python" >&2
    echo "   Run scripts/install.sh first." >&2
    exit 1
fi

# Locate the Hermes workspace.
HERMES_HOME="${HERMES_HOME:-$HOME/.hermes/memories}"
if [ ! -d "$HERMES_HOME" ]; then
    echo "❌ Hermes workspace not found at: $HERMES_HOME" >&2
    echo "   Set \$HERMES_HOME or pass a path." >&2
    exit 1
fi

# Count files for the status line.
COUNT=$(find "$HERMES_HOME" -name "*.md" -o -name "*.markdown" 2>/dev/null | wc -l | tr -d ' ')
echo "🧠 DuckBot brain bootstrap (Hermes Agent)"
echo "   Workspace: $HERMES_HOME"
echo "   Markdown files: $COUNT"
echo

# 1. Doctor
echo "→ Verifying setup..."
"$PY" -m src.cli doctor >/dev/null

# 2. Ingest every .md / .markdown file
echo "→ Ingesting markdown into brain (working tier)..."
"$PY" -m src.cli ingest "$HERMES_HOME" || {
    echo "❌ Ingest failed" >&2
    exit 1
}

# 3. Inflate so Hermes has consolidated context files
echo "→ Inflating consolidated context (MEMORY.md, USER.md, SOUL.md)..."
"$PY" -m src.cli sync --target hermes || {
    echo "⚠ brain_sync failed (non-fatal)" >&2
}

# 4. Done
echo
echo "✓ Bootstrap complete."
echo
echo "Next: register the brain as an MCP server with Hermes:"
echo
echo "    hermes mcp add duckbot-memory \\"
echo "      --command \"$REPO_ROOT/scripts/duckbot-memory-mcp.sh\""
echo
echo "Then add the pre-flight hook so the brain loads at every session"
echo "start (one-call context load with brain_wake_up):"
echo
echo "    Add to ~/.hermesrc or your SessionStart hook:"
echo "    $REPO_ROOT/scripts/hermes-preflight.sh"
echo
echo "    And the post-flight hook to consolidate every session:"
echo "    $REPO_ROOT/scripts/hermes-postflight.sh"