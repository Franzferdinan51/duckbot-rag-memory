#!/usr/bin/env bash
# scripts/install.sh — cross-platform-ish bootstrap (POSIX shell).
#
# duckbot-secret-scan: allowlist-file
#
# This is the "vagrant" installer: it just sets up the venv and deps.
# For OS-specific service integration (auto-restart on boot/crash), use:
#   - macOS:   scripts/install-macos.sh  (launchd plist)
#   - Linux:   scripts/install-linux.sh  (systemd user unit)
#   - Windows: scripts/install.ps1       (Task Scheduler)
#
# Usage (from repo root in any POSIX shell):
#   ./scripts/install.sh
#
# What it does:
#   1. Creates .venv (if missing)
#   2. Installs deps from requirements.txt
#   3. Copies .env.example → .env (if .env missing)
#   4. Makes all scripts in scripts/ executable
#   5. Initializes git (if not already a repo)
#   6. Prints next-steps for your OS

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=== DuckBot RAG + Memory install (POSIX) ==="
echo "Repo: $REPO_ROOT"
echo "OS:   $(uname -s 2>/dev/null || echo unknown)"

# 1. venv
if [[ ! -d .venv ]]; then
  echo "Creating venv..."
  python3 -m venv .venv
fi

# 2. install deps
echo "Installing deps..."
.venv/bin/pip install --quiet --upgrade pip
if [[ -f requirements.txt ]]; then
  .venv/bin/pip install --quiet -r requirements.txt
else
  echo "⚠ No requirements.txt; skipping pip install"
fi

# 3. .env
if [[ ! -f .env ]] && [[ -f .env.example ]]; then
  cp .env.example .env
  echo "Created .env from template. EDIT IT to set LMSTUDIO_URL, LMSTUDIO_KEY, MINIMAX_API_KEY."
elif [[ -f .env ]]; then
  echo ".env already exists"
else
  echo "⚠ No .env or .env.example; skipping"
fi

# 4. make scripts executable
chmod +x scripts/*.sh 2>/dev/null || true

# 5. git init if needed
if [[ ! -d .git ]]; then
  git init
  git add -A
  git commit -m "init: duckbot-rag-memory"
  echo "Initialized git repo. Push with: git remote add origin <url> && git push -u origin main"
else
  echo "Git repo already initialized"
fi

# 6. OS-specific next steps
echo ""
echo "=== Install complete ==="
echo ""
case "$(uname -s 2>/dev/null || echo unknown)" in
  Darwin)
    echo "Next steps (macOS):"
    echo "  1. Edit .env to set LMSTUDIO_URL + LMSTUDIO_KEY (and optional MiniMax key for fallback)"
    echo "  2. ./.venv/bin/python -m src.cli doctor                    # verify all green"
    echo "  3. ./.venv/bin/python -m src.watcher once                  # cold-start full sync"
    echo "  4. ./.venv/bin/python -m src.watcher daemon                # start in background"
    echo "  5. (optional) ./scripts/install-macos.sh                   # auto-restart on boot/crash"
    ;;
  Linux)
    echo "Next steps (Linux):"
    echo "  1. Edit .env to set LMSTUDIO_URL + LMSTUDIO_KEY (and optional MiniMax key for fallback)"
    echo "  2. ./.venv/bin/python -m src.cli doctor                    # verify all green"
    echo "  3. ./.venv/bin/python -m src.watcher once                  # cold-start full sync"
    echo "  4. ./.venv/bin/python -m src.watcher daemon                # start in background"
    echo "  5. (optional) ./scripts/install-linux.sh                    # systemd user unit"
    ;;
  *)
    echo "Next steps (POSIX unknown):"
    echo "  1. Edit .env to set LMSTUDIO_URL + LMSTUDIO_KEY"
    echo "  2. ./.venv/bin/python -m src.cli doctor"
    echo "  3. ./.venv/bin/python -m src.watcher once"
    ;;
esac
echo ""
echo "For Windows, see scripts/install.ps1 and docs in README.md."
