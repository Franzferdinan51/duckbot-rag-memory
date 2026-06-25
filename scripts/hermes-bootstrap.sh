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
echo "→ Step 1/5: Verifying setup..."
"$PY" -m src.cli doctor >/dev/null

# 2. Ingest every .md / .markdown file
echo "→ Step 2/5: Ingesting markdown into brain (working tier)..."
"$PY" -m src.cli ingest "$HERMES_HOME" || {
    echo "❌ Ingest failed" >&2
    exit 1
}

# 3. Inflate so Hermes has consolidated context files
echo "→ Step 3/5: Inflating consolidated context (MEMORY.md, USER.md, SOUL.md)..."
"$PY" -m src.cli sync --target hermes || {
    echo "⚠ brain_sync failed (non-fatal)" >&2
}

# 4. Done
echo
echo "✓ Bootstrap complete."
echo

# Auto-install the Hermes plugin symlink so Hermes's plugin loader finds
# us on next session start. Idempotent: re-running is a no-op.
HERMES_PLUGINS_DIR="${HERMES_HOME%/memories}/plugins/memory/duckbot_brain"
PLUGIN_SRC_INIT="$REPO_ROOT/src/plugins/memory/duckbot_brain/__init__.py"
PLUGIN_SRC_YAML="$REPO_ROOT/src/plugins/memory/duckbot_brain/plugin.yaml"
if [ -f "$PLUGIN_SRC_INIT" ]; then
    mkdir -p "$HERMES_PLUGINS_DIR"
    # Copy (not symlink) the Python module so the plugin loader's import
    # machinery picks it up — Hermes imports plugin packages, it doesn't
    # follow symlinks in all configurations.
    cp "$PLUGIN_SRC_INIT" "$HERMES_PLUGINS_DIR/__init__.py" 2>/dev/null && \
        cp "$PLUGIN_SRC_YAML" "$HERMES_PLUGINS_DIR/plugin.yaml" 2>/dev/null && \
        echo "✓ Plugin installed: $HERMES_PLUGINS_DIR/"
fi

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

# 4. Install the pre-commit secret-scan hook (defense in depth: catches
#    accidental .env / API key commits).
echo
echo "→ Step 4/5: Installing pre-commit secret-scan hook..."
HOOK_SRC="$REPO_ROOT/scripts/secret-scan.sh"
HOOK_DST="$REPO_ROOT/.git/hooks/pre-commit"
if [ -f "$HOOK_SRC" ] && [ -d "$REPO_ROOT/.git" ]; then
    cp "$HOOK_SRC" "$HOOK_DST"
    chmod +x "$HOOK_DST"
    echo "    Installed: $HOOK_DST"
else
    echo "    ⚠ secret-scan.sh not found or no .git; skipping hook install"
fi

# 5. End-to-end demo
echo
echo "→ Step 5/5: Running the end-to-end demo..."
echo
"$REPO_ROOT/scripts/demo.sh" || {
    echo "⚠ demo run failed (non-fatal — try it manually: scripts/demo.sh)" >&2
}