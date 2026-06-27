#!/bin/bash
# demo.command — Run the DuckBot demo (macOS Finder double-click).
# See demo.sh for the equivalent that runs in any Terminal.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
info()  { echo -e "${GREEN}  →${RESET} $1"; }
warn()  { echo -e "${YELLOW}  ⚠${RESET} $1"; }
ok()    { echo -e "${GREEN}  ✓${RESET} $1"; }
error() { echo -e "${RED}  ✗${RESET} $1"; }

echo ""
echo -e "${BOLD}🧠  DuckBot RAG + Memory — Demo${RESET}"
echo ""

# Load .env
if [[ -f .env ]]; then
    set -a
    . <(grep -vE '^\s*#' .env | grep -vE '^\s*$')
    set +a
fi

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

# Doctor
info "Verifying setup..."
if ! "$PYTHON" -m src.cli doctor 2>&1 | grep -q "✗"; then
    ok "All checks passed"
else
    warn "Some checks failed. Run ./scripts/setup.command to fix."
fi
echo ""

# Demo steps
echo -e "${CYAN}[1/4]${RESET} ${BOLD}Seeding demo corpus (idempotent)...${RESET}"
"$PYTHON" -m src.cli seed-demo 2>&1 | tail -5
echo ""

echo -e "${CYAN}[2/4]${RESET} ${BOLD}Wake-up (session-start context)...${RESET}"
"$PYTHON" -m src.cli wake-up 2>&1 | head -30
echo "  ...(truncated; use --json for full output)"
echo ""

echo -e "${CYAN}[3/4]${RESET} ${BOLD}Querying: \"How do I restart the BATMAN container?\"${RESET}"
"$PYTHON" -m src.cli query "How do I restart the BATMAN container?" -n 3 2>&1
echo ""

echo -e "${CYAN}[4/4]${RESET} ${BOLD}Querying: \"What are DuckBot's design constraints?\"${RESET}"
"$PYTHON" -m src.cli query "What are DuckBot's design constraints?" -n 3 2>&1
echo ""

echo -e "${BOLD}✅ Demo complete!${RESET}"
echo ""
echo "Next:"
echo "  ./scripts/duckbot-ask \"your question\""
echo "  ./scripts/start.command    # start watcher daemon (recommended)"
echo "  ./scripts/openclaw-bootstrap.sh  # set up with OpenClaw"
echo "  ./scripts/hermes-bootstrap.sh   # set up with Hermes Agent"
echo ""
echo "Press Enter to close..."
read -p "" _
