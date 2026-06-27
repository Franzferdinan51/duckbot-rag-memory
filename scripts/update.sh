#!/usr/bin/env bash
# update.sh — Update DuckBot to the latest version (Linux / macOS Terminal).
#
# Usage (from repo root):
#   ./scripts/update.sh              # full update + deps + doctor
#   ./scripts/update.sh --dry-run   # check if updates are available
#   ./scripts/update.sh --no-deps   # skip pip install
#   ./scripts/update.sh --no-doctor # skip doctor check
#
# For agents / machine use (returns JSON):
#   python -m src.cli update --dry-run
#   python -m src.cli update
#   python -m src.cli update --no-deps --no-doctor
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

if ! command -v git >/dev/null 2>&1; then
    error "git is not installed"
    exit 1
fi

if [[ -x .venv/bin/python ]]; then
    PYTHON=".venv/bin/python"
elif [[ -x .venv/Scripts/python.exe ]]; then
    PYTHON=".venv/Scripts/python.exe"
else
    error "No venv found. Run ./scripts/setup.sh first."
    exit 1
fi

N=$((N + 1))
step "Step $N: Run update (python -m src.cli update $*)"

info "Calling: $PYTHON -m src.cli update $*"
echo ""

UPDATE_OUTPUT=$("$PYTHON" -m src.cli update "$@" 2>&1) || true
echo "$UPDATE_OUTPUT"

# Parse JSON result for pretty summary
IS_JSON=false
if echo "$UPDATE_OUTPUT" | python3 -c "import sys,json; json.load(sys.stdin); sys.exit(0)" 2>/dev/null; then
    IS_JSON=true
fi

if [[ "$IS_JSON" == "true" ]]; then
    RESULT=$(echo "$UPDATE_OUTPUT" | python3 -c "
import sys,json
d=json.load(sys.stdin)
print(json.dumps({
    'was_updated': d.get('was_updated','unknown'),
    'commits_behind': d.get('commits_behind','unknown'),
    'doctor_passed': d.get('doctor_passed','unknown'),
    'error': d.get('error',''),
    'had_local_changes': d.get('had_local_changes','unknown'),
}, default=str))
" 2>/dev/null || echo '{"was_updated":"unknown"}')

    WAS_UPDATED=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('was_updated','unknown'))")
    BEHIND=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('commits_behind','unknown'))")
    DOCTOR=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('doctor_passed','unknown'))")
    ERROR=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error',''))")

    if [[ -n "$ERROR" && "$ERROR" != "null" ]]; then
        if [[ "$ERROR" == "not a git repo" ]]; then
            error "Not in a git repository."
        else
            warn "Error: $ERROR"
        fi
    elif [[ "$WAS_UPDATED" == "True" ]]; then
        if [[ "$DOCTOR" == "True" ]]; then
            ok "Updated successfully — doctor passed"
        elif [[ "$DOCTOR" == "False" ]]; then
            warn "Updated — some doctor checks failed (see above)"
        else
            ok "Updated successfully"
        fi
    elif [[ "$WAS_UPDATED" == "False" ]]; then
        if [[ "$BEHIND" == "0" ]]; then
            ok "Already up to date"
        else
            warn "Update failed — check the output above"
        fi
    fi
else
    warn "Non-JSON output returned — check above for errors"
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
echo "For agent/machine use (returns JSON):"
echo "    $PYTHON -m src.cli update --dry-run"
echo "    $PYTHON -m src.cli update"
echo ""
