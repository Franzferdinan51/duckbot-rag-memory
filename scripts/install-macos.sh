#!/usr/bin/env bash
# scripts/install.sh — bootstrap the project + wire the watcher to launchd.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=== DuckBot RAG + Memory install ==="
echo "Repo: $REPO_ROOT"

# 1. venv
if [[ ! -d .venv ]]; then
  echo "Creating venv..."
  python3 -m venv .venv
fi

# 2. install deps
echo "Installing deps..."
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

# 3. .env
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from template. EDIT IT to set LMSTUDIO_URL, LMSTUDIO_KEY, LMSTUDIO_MODEL, LMSTUDIO_RERANK_MODEL, MINIMAX_API_KEY."
else
  echo ".env already exists"
fi

# 4. make scripts executable. The `*.sh` glob misses `duckbot-ask`
# (no extension) and the Python helpers, so chmod them explicitly.
chmod +x scripts/*.sh scripts/duckbot-ask 2>/dev/null || true
chmod +x scripts/_format_*.py 2>/dev/null || true

# 5. git init if needed
if [[ ! -d .git ]]; then
  git init
  git add -A
  git commit -m "init: duckbot-rag-memory v0.2.0"
  echo "Initialized git repo. Push with: git remote add origin <url> && git push -u origin main"
else
  echo "Git repo already initialized"
fi

# 6. Wire the watcher to launchd (auto-restart on crash + on boot)
echo ""
echo "macOS launchd setup (auto-restart on crash + on boot):"
echo "  cp scripts/com.duckbot.memory-watcher.plist ~/Library/LaunchAgents/"
echo "  launchctl load -w ~/Library/LaunchAgents/com.duckbot.memory-watcher.plist"
echo ""

if [[ "$(uname)" == "Darwin" ]]; then
  read -p "Install the launchd plist now? [y/N] " install_launchd
  if [[ "$install_launchd" == "y" || "$install_launchd" == "Y" ]]; then
    PLIST_SRC="$REPO_ROOT/scripts/com.duckbot.memory-watcher.plist"
    PLIST_DST="$HOME/Library/LaunchAgents/com.duckbot.memory-watcher.plist"
    # Template-substitute __REPO_ROOT__ with the actual repo path.
    # The plist is committed as a template (no hardcoded paths) so it
    # works for any user who clones the repo.
    mkdir -p "$(dirname "$PLIST_DST")"
    sed "s|__REPO_ROOT__|$REPO_ROOT|g" "$PLIST_SRC" > "$PLIST_DST"
    chmod 644 "$PLIST_DST"
    launchctl unload "$PLIST_DST" 2>/dev/null || true
    launchctl load -w "$PLIST_DST"
    echo "Installed and started: $PLIST_DST"
    echo "Manage with:"
    echo "  launchctl list | grep duckbot.memory"
    echo "  launchctl unload ~/Library/LaunchAgents/com.duckbot.memory-watcher.plist  # to stop"
  fi
fi

echo ""
echo "=== Install complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env to set LMSTUDIO_URL + LMSTUDIO_KEY + LMSTUDIO_MODEL + LMSTUDIO_RERANK_MODEL (and optional MiniMax key for fallback)"
echo "  2. Run: .venv/bin/python -m src.cli doctor         (verify all green)"
echo "  3. Run: .venv/bin/python -m src.watcher once       (cold-start full sync)"
echo "  4. Run: .venv/bin/python -m src.watcher status     (should show: running)"
echo "  5. Query: .venv/bin/python -m src.cli query 'What did we decide about cloud-only models?'"
echo "  6. Push to GitHub: git remote add origin <url> && git push -u origin main"
