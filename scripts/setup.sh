#!/usr/bin/env bash
# setup.sh — DuckBot one-click setup for Linux and macOS (Terminal).
#
# Usage (from repo root):
#   ./scripts/setup.sh
#
# What it does:
#   1. Create .venv
#   2. Install deps
#   3. Copy .env.example → .env (if missing)
#   4. Run doctor to verify
#   5. Seed demo corpus
#   6. Run a sample query
#   7. Print next steps
#
# For macOS Finder, double-click setup.command instead.

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
step()  { echo ""; echo -e "${CYAN}[${STEP}]${RESET} ${BOLD}$1${RESET}"; }

echo ""
echo -e "${BOLD}🧠  DuckBot RAG + Memory — One-Click Setup${RESET}"
echo -e "    Repo: $REPO_ROOT"
echo -e "    OS:   $(uname -s)"

STEP=1
step "Step $STEP: Create Python virtual environment"

if [[ -d .venv ]]; then
    info "venv already exists at .venv"
else
    info "Creating .venv..."
    python3 -m venv .venv || {
        error "Failed to create venv. Is Python 3.9+ installed?"
        exit 1
    }
    ok "venv created"
fi

if [[ -x .venv/bin/python ]]; then
    PYTHON=".venv/bin/python"
elif [[ -x .venv/Scripts/python.exe ]]; then
    PYTHON=".venv/Scripts/python.exe"
else
    error "Python not found in .venv"
    exit 1
fi

STEP=$((STEP + 1))
step "Step $STEP: Install Python dependencies"

info "Upgrading pip..."
"$PYTHON" -m pip install --quiet --upgrade pip 2>/dev/null || true

if [[ -f requirements.txt ]]; then
    info "Installing from requirements.txt..."
    "$PYTHON" -m pip install -r requirements.txt
    ok "Dependencies installed"
else
    error "requirements.txt not found"
    exit 1
fi

STEP=$((STEP + 1))
step "Step $STEP: Configure environment"

if [[ ! -f .env ]] && [[ -f .env.example ]]; then
    cp .env.example .env
    ok "Created .env from template"
    warn "Edit .env to set your API keys (see .env.example for options)"
else
    ok ".env already exists"
fi

# Load .env for doctor and seed
if [[ -f .env ]]; then
    set -a
    . <(grep -vE '^\s*#' .env | grep -vE '^\s*$')
    set +a
fi

STEP=$((STEP + 1))
step "Step $STEP: Verify setup"

# Ensure scripts are executable
chmod +x scripts/*.sh scripts/duckbot-ask scripts/setup.command scripts/demo.command scripts/start.command 2>/dev/null || true
chmod +x scripts/_format_*.py 2>/dev/null || true

# Check LM Studio
if curl -s --max-time 2 "${LMSTUDIO_URL:-http://127.0.0.1:1234/v1/}/models" > /dev/null 2>&1; then
    ok "LM Studio is running at ${LMSTUDIO_URL:-http://127.0.0.1:1234/v1/}"
else
    warn "LM Studio not reachable at ${LMSTUDIO_URL:-http://127.0.0.1:1234/v1/}"
    warn "If you use LM Studio locally, load text-embedding-embeddinggemma-300m and qwen3-reranker-0.6b first"
    warn "Or edit .env to use MINIMAX_API_KEY or DUCKBOT_EMBEDDING=local"
fi

info "Running doctor..."
if "$PYTHON" -m src.cli doctor 2>&1 | tee /dev/stderr | grep -q "✗"; then
    warn "Some doctor checks failed — see above. You may need to edit .env."
else
    ok "All checks passed"
fi

STEP=$((STEP + 1))
step "Step $STEP: Seed demo data"

SEED_OUTPUT=$("$PYTHON" -m src.cli seed-demo 2>&1)
echo "$SEED_OUTPUT"
if echo "$SEED_OUTPUT" | grep -q '"stored": [1-9]'; then
    ok "Demo seeded successfully"
elif echo "$SEED_OUTPUT" | grep -q '"stored": 0'; then
    warn "Demo already seeded (idempotent — this is fine)"
else
    warn "Seed result unclear"
fi

STEP=$((STEP + 1))
step "Step $STEP: Try a query"

info "Asking: \"How do I restart the BATMAN container?\""
"$PYTHON" -m src.cli query "How do I restart the BATMAN container?" -n 3 2>&1

echo ""
echo "─────────────────────────────────────────────────────────"
echo -e "  ${BOLD}✅ Setup complete!${RESET}"
echo "─────────────────────────────────────────────────────────"
echo ""
echo "Next steps:"
echo ""
echo -e "  ${BOLD}Start the watcher daemon (recommended):${RESET}"
echo "    ./scripts/start.sh              # Linux / macOS Terminal"
echo "    ./scripts/start.command         # macOS Finder (double-click)"
echo ""
echo -e "  ${BOLD}Query the brain:${RESET}"
echo "    ./scripts/duckbot-ask \"your question here\""
echo "    ./scripts/duckbot-ask -f snippet \"BATMAN restart steps\""
echo ""
echo -e "  ${BOLD}Run the demo again:${RESET}"
echo "    ./scripts/demo.sh              # Linux / macOS Terminal"
echo "    ./scripts/demo.command         # macOS Finder (double-click)"
echo ""
echo -e "  ${BOLD}Set up with OpenClaw:${RESET}"
echo "    ./scripts/openclaw-bootstrap.sh"
echo ""
echo -e "  ${BOLD}Set up with Hermes Agent:${RESET}"
echo "    ./scripts/hermes-bootstrap.sh"
echo ""
echo -e "  ${BOLD}Register as MCP server (Claude Code, Cursor, etc.):${RESET}"
echo "    hermes mcp add duckbot-memory --command \"$(pwd)/scripts/duckbot-memory-mcp.sh\""
echo ""
echo -e "  ${BOLD}Edit your .env${RESET} to configure embedding provider and API keys."
echo "     See README.md or INSTALL.md for the full guide."
echo ""
