#!/usr/bin/env bash
# scripts/openclaw-bootstrap.sh — one-command OpenClaw → brain setup.
#
# This script dramatically expands OpenClaw's default memory by ingesting
# every markdown file in its workspace into the brain in one pass, then
# registering the brain as an MCP server. After running this once, every
# OpenClaw session has the full corpus available via brain_wake_up.
#
# What it does:
#   1. Verify venv + CLI work
#   2. Discover OpenClaw workspace (default ~/.openclaw/workspace/)
#   3. Ingest every .md file into the brain (working, episodic, etc.)
#   4. Run brain_inflate so OpenClaw has consolidated context files
#   5. Print the MCP registration command for OpenClaw
#
# Idempotent: re-running on the same files is a no-op (content-hash dedup).
#
# Usage:
#   ./scripts/openclaw-bootstrap.sh                     # default workspace
#   ./scripts/openclaw-bootstrap.sh /custom/openclaw/path  # custom path
#   OPENCLAW_HOME=/custom/path ./scripts/openclaw-bootstrap.sh  # env override

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

# Locate the OpenClaw workspace.
OPENCLAW_HOME="${1:-${OPENCLAW_HOME:-$HOME/.openclaw/workspace}}"
if [ ! -d "$OPENCLAW_HOME" ]; then
    echo "❌ OpenClaw workspace not found at: $OPENCLAW_HOME" >&2
    echo "   Pass the path as an argument or set \$OPENCLAW_HOME." >&2
    exit 1
fi

# Count files for the status line.
COUNT=$(find "$OPENCLAW_HOME" -name "*.md" -o -name "*.markdown" 2>/dev/null | wc -l | tr -d ' ')
echo "🧠 DuckBot brain bootstrap"
echo "   Workspace: $OPENCLAW_HOME"
echo "   Markdown files: $COUNT"
echo

# 1. Doctor
echo "→ Verifying setup..."
"$PY" -m src.cli doctor >/dev/null

# 2. Ingest every .md / .markdown file
echo "→ Ingesting markdown into brain (working tier, then semantic)..."
"$PY" -m src.cli ingest "$OPENCLAW_HOME" || {
    echo "❌ Ingest failed" >&2
    exit 1
}

# 3. Inflate the brain so OpenClaw's MEMORY.md/USER.md/SOUL.md are fresh
echo "→ Inflating consolidated context (MEMORY.md, USER.md, SOUL.md)..."
"$PY" -m src.cli sync --target openclaw || {
    echo "⚠ brain_sync failed (non-fatal)" >&2
}

# 4. Done
echo
echo "✓ Bootstrap complete."
echo
echo "Next: register the brain as an MCP server with OpenClaw:"
echo
echo "    Edit ~/.openclaw/openclaw.json and add under mcp.servers:"
echo "    {"
echo "      \"duckbot-brain\": {"
echo "        \"command\": \"$PY\","
echo "        \"args\": [\"-m\", \"src.mcp_server\"]"
echo "      }"
echo "    }"
echo
echo "    Or use the helper script:"
echo "    $REPO_ROOT/scripts/duckbot-memory-mcp.sh"
echo
echo "Then in OpenClaw, call brain_wake_up at session start to load"
echo "the full corpus (memories + blocks + graph + FSRS review queue)"
echo "in one MCP call. See README § Enhanced Brain for details."