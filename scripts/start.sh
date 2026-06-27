#!/usr/bin/env bash
# start.sh — Start the DuckBot memory watcher daemon (Linux / macOS Terminal).
#
# Usage (from repo root):
#   ./scripts/start.sh
#
# Double-click on macOS? Use start.command instead.
#
# The watcher polls every 5 minutes, dedups by content hash, and logs to
# data/watcher.log. Use scripts/install-macos.sh (launchd) or
# scripts/install-linux.sh (systemd) for auto-restart on boot/crash.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; RESET='\033[0m'
info()  { echo -e "${GREEN}  →${RESET} $1"; }
warn()  { echo -e "${YELLOW}  ⚠${RESET} $1"; }
ok()    { echo -e "${GREEN}  ✓${RESET} $1"; }
error() { echo -e "${RED}  ✗${RESET} $1"; }

echo ""
echo -e "${BOLD}🧠  DuckBot — Starting watcher daemon${RESET}"
echo ""

# Detect python
if [[ -x .venv/bin/python ]]; then
    PYTHON=".venv/bin/python"
elif [[ -x .venv/Scripts/python.exe ]]; then
    PYTHON=".venv/Scripts/python.exe"
else
    error "No venv found. Run ./scripts/setup.sh first."
    exit 1
fi

# Load .env
if [[ -f .env ]]; then
    set -a
    . <(grep -vE '^\s*#' .env | grep -vE '^\s*$')
    set +a
fi

: "${OPENCLAW_MEMORY:=$HOME/.openclaw/workspace/memory}"
: "${OPENCLAW_WORKSPACE:=$HOME/.openclaw/workspace}"

# Check if already running
STATUS=$("$PYTHON" -m src.watcher status 2>&1) || true
if echo "$STATUS" | grep -q "running"; then
    ok "Watcher is already running"
    echo "  $STATUS"
    exit 0
fi

# Build watch paths
WATCH_ARGS=()
if [[ -n "${OPENCLAW_MEMORY:-}" && -e "${OPENCLAW_MEMORY}" ]]; then
    WATCH_ARGS+=("${OPENCLAW_MEMORY}")
fi
if [[ -n "${OPENCLAW_WORKSPACE:-}" && -e "${OPENCLAW_WORKSPACE}" ]]; then
    for f in AGENTS.md SOUL.md USER.md IDENTITY.md TOOLS.md MEMORY.md README.md; do
        [[ -f "$OPENCLAW_WORKSPACE/$f" ]] && WATCH_ARGS+=("$OPENCLAW_WORKSPACE/$f")
    done
fi
for f in AGENTS.md SOUL.md USER.md IDENTITY.md TOOLS.md README.md; do
    [[ -f "$REPO_ROOT/$f" ]] && WATCH_ARGS+=("$REPO_ROOT/$f")
done

mkdir -p data
rm -f data/watcher.pid

info "Starting watcher (polls every 5 min, content-hash dedup)..."
echo "  Watch paths: ${WATCH_ARGS[*]:-defaults}"
echo "  Log: $REPO_ROOT/data/watcher.log"
echo ""

nohup "$PYTHON" -m src.watcher run "${WATCH_ARGS[@]}" \
    </dev/null >>"$REPO_ROOT/data/watcher.log" 2>&1 &
WPID=$!
disown $WPID 2>/dev/null || true

sleep 2
STATUS=$("$PYTHON" -m src.watcher status 2>&1) || true
if echo "$STATUS" | grep -q "running"; then
    ok "Watcher started"
    echo "  PID: $WPID"
    echo "  Status: $STATUS"
else
    warn "Watcher may not have started cleanly. Check:"
    echo "  $("$PYTHON" -m src.watcher status 2>&1)"
    echo "  tail -20 $REPO_ROOT/data/watcher.log"
fi

echo ""
echo "Manage:"
echo "  ${PYTHON} -m src.watcher status"
echo "  ${PYTHON} -m src.watcher stop"
echo "  tail -f ${REPO_ROOT}/data/watcher.log"
echo ""
echo "For auto-restart on boot/crash:"
echo "  macOS:  ./scripts/install-macos.sh"
echo "  Linux:  ./scripts/install-linux.sh"
echo ""
