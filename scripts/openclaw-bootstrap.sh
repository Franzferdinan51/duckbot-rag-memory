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
echo "→ Step 1/5: Verifying setup..."
"$PY" -m src.cli doctor >/dev/null

# 2. Ingest every .md / .markdown file
echo "→ Step 2/5: Ingesting markdown into brain (working tier, then semantic)..."
"$PY" -m src.cli ingest "$OPENCLAW_HOME" || {
    echo "❌ Ingest failed" >&2
    exit 1
}

# 3. Inflate the brain so OpenClaw's MEMORY.md/USER.md/SOUL.md are fresh
echo "→ Step 3/5: Inflating consolidated context (MEMORY.md, USER.md, SOUL.md)..."
"$PY" -m src.cli sync --target openclaw || {
    echo "⚠ brain_sync failed (non-fatal)" >&2
}

# 4. Done
echo
echo "✓ Bootstrap complete."
echo

# Auto-install the duckbot-brain skill into the OpenClaw workspace so
# agents discover it on next session start. Idempotent: re-running on an
# existing symlink is a no-op.
SKILL_DST_DIR="$HOME/.openclaw/workspace/skills/duckbot-brain"
SKILL_SRC="$REPO_ROOT/skills/duckbot-brain/SKILL.md"
if [ -f "$SKILL_SRC" ]; then
    mkdir -p "$SKILL_DST_DIR"
    if ln -sf "$SKILL_SRC" "$SKILL_DST_DIR/SKILL.md" 2>/dev/null; then
        echo "✓ Skill installed: $SKILL_DST_DIR/SKILL.md → $SKILL_SRC"
    else
        # Fallback: copy if symlinks aren't supported (some Windows mounts).
        cp "$SKILL_SRC" "$SKILL_DST_DIR/SKILL.md" && \
            echo "✓ Skill copied: $SKILL_DST_DIR/SKILL.md"
    fi
fi

# Install the native OpenClaw plugin (extensions/duckbot-memory/) so
# brain_wake_up auto-fires on session_start and brain_sync on session_end.
# The plugin is a pure Node.js shim — zero npm deps — that spawns the
# Python MCP server as a subprocess and proxies 67 tools + session hooks.
# Idempotent: re-running replaces the symlink.
OPENCLAW_PLUGINS_DIR="${OPENCLAW_HOME%/workspace}/extensions/duckbot-memory"
PLUGIN_SRC="$REPO_ROOT/extensions/duckbot-memory"
if [ -d "$PLUGIN_SRC" ]; then
    mkdir -p "$(dirname "$OPENCLAW_PLUGINS_DIR")"
    rm -rf "$OPENCLAW_PLUGINS_DIR" 2>/dev/null || true
    if ln -sf "$PLUGIN_SRC" "$OPENCLAW_PLUGINS_DIR" 2>/dev/null; then
        echo "✓ OpenClaw plugin symlinked: $OPENCLAW_PLUGINS_DIR → $PLUGIN_SRC"
    else
        # Fallback: copy (Windows mounts often reject symlinks).
        cp -R "$PLUGIN_SRC" "$OPENCLAW_PLUGINS_DIR" && \
            echo "✓ OpenClaw plugin copied: $OPENCLAW_PLUGINS_DIR"
    fi
    echo "  Active after: openclaw gateway restart"
    echo "  Verify with:  openclaw plugins list | grep duckbot-memory"
else
    echo "  ⚠ Plugin source missing at $PLUGIN_SRC — skipping native install"
fi

echo
echo "Next: restart the OpenClaw gateway so it loads the plugin."
echo "    openclaw gateway restart"
echo
echo "What the plugin gives you:"
echo "  ✓ 67 brain tools registered natively (brain_wake_up / brain_recall / ...)"
echo "  ✓ session_start hook auto-fires brain_wake_up — context loads without"
echo "    the agent having to remember to call it"
echo "  ✓ session_end hook auto-fires brain_sync — high-importance facts"
echo "    get written back to OpenClaw's MEMORY.md / USER.md / SOUL.md"
echo
echo "Non-OpenClaw clients (Claude Code / Cursor / Codex) use the generic"
echo "JSON-RPC adapter at src/extensions/duckbot_brain/adapter.py — see"
echo "extensions/duckbot-memory/README.md for the per-client JSON snippet."

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
echo "→ Step 5/5: running the end-to-end demo..."
echo
"$REPO_ROOT/scripts/demo.sh" || {
    echo "⚠ demo run failed (non-fatal — try it manually: scripts/demo.sh)" >&2
}
