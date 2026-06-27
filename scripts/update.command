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

N=1
step "Step $N: Check prerequisites"

if [[ -x .venv/bin/python ]]; then
    PYTHON=".venv/bin/python"
elif [[ -x .venv/Scripts/python.exe ]]; then
    PYTHON=".venv/Scripts/python.exe"
else
    error "No venv found. Run ./scripts/setup.command first."
    read -p "Press Enter to exit..." _
    exit 1
fi

N=$((N + 1))
step "Step $N: Run update"

info "Calling: python -m src.cli update"
echo ""

UPDATE_OUTPUT=$("$PYTHON" -m src.cli update "$@" 2>&1) || true
echo "$UPDATE_OUTPUT"

if echo "$UPDATE_OUTPUT" | python3 -c "import sys,json; json.load(sys.stdin); sys.exit(0)" 2>/dev/null; then
    WAS_UPDATED=$(echo "$UPDATE_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('was_updated', 'unknown'))" 2>/dev/null || echo "unknown")
    BEHIND=$(echo "$UPDATE_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('commits_behind', 'unknown'))" 2>/dev/null || echo "unknown")
    DOCTOR=$(echo "$UPDATE_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('doctor_passed', 'unknown'))" 2>/dev/null || echo "unknown")
    ERROR=$(echo "$UPDATE_OUTPUT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error', ''))" 2>/dev/null || echo "")

    if [[ "$ERROR" == "not a git repo" ]]; then
        error "Not in a git repository."
    elif [[ "$ERROR" == "no remote" ]]; then
        warn "No remote configured."
    elif [[ "$WAS_UPDATED" == "True" ]]; then
        if [[ "$DOCTOR" == "True" ]]; then
            ok "Updated — doctor passed"
        else
            warn "Updated — some doctor checks failed (see above)"
        fi
    elif [[ "$WAS_UPDATED" == "False" ]]; then
        if [[ "$BEHIND" == "0" ]]; then
            ok "Already up to date"
        else
            warn "Update failed — check above"
        fi
    fi
fi

echo ""
echo "─────────────────────────────────────────────────────────"
echo -e "  ${BOLD}✅ Update complete!${RESET}"
echo "─────────────────────────────────────────────────────────"
echo ""
echo "Run the demo:    ./scripts/demo.command"
echo "Query the brain: ./scripts/duckbot-ask"
echo ""
echo "For agents / scripts (JSON output):"
echo "  .venv/bin/python -m src.cli update"
echo ""
echo "Press Enter to close..."
read -p "" _
