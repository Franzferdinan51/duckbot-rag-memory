#!/bin/bash
# update.command — Update DuckBot to the latest version (macOS Finder double-click).
# See update.sh for the equivalent that runs in any Terminal.

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
echo ""

N=1
step "Step $N: Check git status"

# Detect python
if [[ -x .venv/bin/python ]]; then
    PYTHON=".venv/bin/python"
elif [[ -x .venv/Scripts/python.exe ]]; then
    PYTHON=".venv/Scripts/python.exe"
else
    error "No venv found. Run ./scripts/setup.command first."
    read -p "Press Enter to exit..." _
    exit 1
fi

# Check for uncommitted changes
if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
    warn "You have uncommitted changes — update will stash them"
fi

# Check remote
if ! git remote get-url origin >/dev/null 2>&1; then
    warn "No remote configured. Skipping git pull."
else
    N=$((N + 1))
    step "Step $N: Pull latest changes"

    info "Fetching latest from origin/main..."
    git fetch origin 2>&1

    BEHIND=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo 0)
    if [[ "$BEHIND" == "0" ]]; then
        ok "Already up to date (origin/main)"
    else
        info "Your branch is behind origin/main by $BEHIND commit(s)"
        git stash 2>/dev/null || true
        git pull --rebase origin main 2>&1
        ok "Updated to $(git log -1 --oneline origin/main)"
    fi
fi

N=$((N + 1))
step "Step $N: Update dependencies"

info "Upgrading pip..."
"$PYTHON" -m pip install --quiet --upgrade pip 2>/dev/null

if [[ -f requirements.txt ]]; then
    info "Installing new dependencies..."
    "$PYTHON" -m pip install -r requirements.txt
    ok "Dependencies up to date"
fi

N=$((N + 1))
step "Step $N: Verify update"

info "Running doctor..."
DOCTOR_OUTPUT=$("$PYTHON" -m src.cli doctor 2>&1)
echo "$DOCTOR_OUTPUT"

if echo "$DOCTOR_OUTPUT" | grep -q "✗"; then
    warn "Some checks failed — see above"
else
    ok "All checks passed"
fi

N=$((N + 1))
step "Step $N: Run doctor checks on existing memory"

info "Checking store integrity..."
STATS=$("$PYTHON" -m src.cli stats 2>&1)
echo "$STATS"

if echo "$STATS" | grep -q '"total": 0'; then
    warn "Store is empty — re-seed with:"
    echo "    .venv/bin/python -m src.cli seed-demo"
else
    ok "Store intact: $(echo "$STATS" | grep -o '"total": [0-9]*' | grep -o '[0-9]*') chunks"
fi

echo ""
echo "─────────────────────────────────────────────────────────"
echo -e "  ${BOLD}✅ Update complete!${RESET}"
echo "─────────────────────────────────────────────────────────"
echo ""
echo "Run the demo:"
echo "    ./scripts/demo.command"
echo ""
echo "Query the brain:"
echo "    ./scripts/duckbot-ask \"your question\""
echo ""
echo "Restart the watcher daemon:"
echo "    ./scripts/start.command"
echo ""
echo "Press Enter to close..."
read -p "" _
