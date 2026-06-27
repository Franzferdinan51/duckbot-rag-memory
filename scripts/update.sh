#!/usr/bin/env bash
# update.sh — Update DuckBot to the latest version (Linux / macOS Terminal).
#
# Usage (from repo root):
#   ./scripts/update.sh
#
# What it does:
#   1. Stash any local changes
#   2. git pull --rebase origin main
#   3. Upgrade pip + reinstall deps from requirements.txt
#   4. Run doctor to verify
#   5. Check store integrity
#
# Double-click on macOS? Use update.command instead.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
info()  { echo -e "${GREEN}  →${RESET} $1"; }
warn()  { echo -e "${YELLOW}  ⚠${RESET} $1"; }
ok()    { echo -e "${GREEN}  ✓${RESET} $1"; }
error() { echo -e "${RED}  ✗${RESET} $1"; }
step()  { echo ""; echo -e "${CYAN}[${N}]${RESET} ${BOLD}$1${RESET}"; }

echo ""
echo -e "${BOLD}🧠  DuckBot RAG + Memory — Update${RESET}"
echo "    Repo: $REPO_ROOT"
echo "    OS:   $(uname -s)"

N=1
step "Step $N: Check prerequisites"

# Detect python
if [[ -x .venv/bin/python ]]; then
    PYTHON=".venv/bin/python"
elif [[ -x .venv/Scripts/python.exe ]]; then
    PYTHON=".venv/Scripts/python.exe"
else
    error "No venv found. Run ./scripts/setup.sh first."
    exit 1
fi

# Check git
if ! command -v git >/dev/null 2>&1; then
    error "git is not installed"
    exit 1
fi

N=$((N + 1))
step "Step $N: Stash local changes"

# Check for uncommitted changes
if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
    warn "Stashing uncommitted changes..."
    git stash 2>/dev/null || true
    ok "Changes stashed"
else
    ok "No uncommitted changes"
fi

N=$((N + 1))
step "Step $N: Pull latest changes"

if ! git remote get-url origin >/dev/null 2>&1; then
    warn "No remote configured. Skipping git pull."
else
    BEHIND=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo 0)
    if [[ "$BEHIND" == "0" ]]; then
        ok "Already up to date (origin/main)"
    else
        info "Behind origin/main by $BEHIND commit(s) — pulling..."
        git fetch origin
        git pull --rebase origin main
        ok "Updated to $(git log -1 --oneline origin/main)"
    fi
fi

N=$((N + 1))
step "Step $N: Update dependencies"

info "Upgrading pip..."
"$PYTHON" -m pip install --quiet --upgrade pip 2>/dev/null || true

if [[ -f requirements.txt ]]; then
    info "Installing updated dependencies..."
    "$PYTHON" -m pip install -r requirements.txt
    ok "Dependencies up to date"
fi

N=$((N + 1))
step "Step $N: Verify update"

info "Running doctor..."
if "$PYTHON" -m src.cli doctor 2>&1 | tee /dev/stderr | grep -q "✗"; then
    warn "Some checks failed — see above"
else
    ok "All checks passed"
fi

N=$((N + 1))
step "Step $N: Check store integrity"

STATS=$("$PYTHON" -m src.cli stats 2>&1)
echo "$STATS"
if echo "$STATS" | grep -q '"total": 0'; then
    warn "Store is empty — re-seed with: $PYTHON -m src.cli seed-demo"
else
    TOTAL=$(echo "$STATS" | grep -o '"total": [0-9]*' | grep -o '[0-9]*')
    ok "Store intact: $TOTAL chunks"
fi

echo ""
echo "─────────────────────────────────────────────────────────"
echo -e "  ${BOLD}✅ Update complete!${RESET}"
echo "─────────────────────────────────────────────────────────"
echo ""
echo "Run the demo:"
echo "    ./scripts/demo.sh"
echo ""
echo "Query the brain:"
echo "    ./scripts/duckbot-ask \"your question\""
echo ""
echo "Restart the watcher daemon:"
echo "    ./scripts/start.sh"
echo ""
