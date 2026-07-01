#!/usr/bin/env bash
# reconnect.sh — Restart the DuckBot brain MCP and reattach to the OpenClaw gateway.
#
# Usage:
#   ~/Desktop/duckbot-rag-memory/scripts/reconnect.sh
#   ~/Desktop/duckbot-rag-memory/scripts/reconnect.sh --force
#
# What it does:
#   1. Find any running src.mcp_server processes
#   2. Kill them
#   3. Tell the OpenClaw gateway to reload (SIGUSR1 or restart)
#   4. Wait for the gateway to respawn the MCP
#   5. Verify with a brain_recall probe
#
# The hard part: the OpenClaw gateway is at pid from pgrep. We don't manage
# it directly — we signal it to restart, which causes it to respawn the MCP.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON="$REPO_ROOT/.venv/bin/python"
elif [[ -x "$REPO_ROOT/.venv/Scripts/python.exe" ]]; then
    PYTHON="$REPO_ROOT/.venv/Scripts/python.exe"
else
    echo "✗ No .venv python at $REPO_ROOT/.venv"
    exit 1
fi

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
info()  { echo -e "${GREEN}  →${RESET} $1"; }
warn()  { echo -e "${YELLOW}  ⚠${RESET} $1"; }
ok()    { echo -e "${GREEN}  ✓${RESET} $1"; }
error() { echo -e "${RED}  ✗${RESET} $1"; }

echo -e "${BOLD}🧠  DuckBot brain — reconnect${RESET}"
echo ""

# Step 1: find running MCP
echo "  → Step 1: Find running MCP processes"
MCP_PIDS=$(pgrep -f "src.mcp_server" 2>/dev/null || true)
if [ -n "$MCP_PIDS" ]; then
    info "Found MCP pids: $MCP_PIDS"
else
    info "No MCP processes running"
fi

# Step 2: find the OpenClaw gateway
echo "  → Step 2: Find OpenClaw gateway"
GATEWAY_PID=$(pgrep -f "openclaw/dist/index.js gateway" 2>/dev/null | head -1 || true)
if [ -z "$GATEWAY_PID" ]; then
    warn "No OpenClaw gateway running. Brain will start but won't be connected."
fi

# Step 3: kill old MCP processes
echo "  → Step 3: Kill old MCP processes"
if [ -n "$MCP_PIDS" ]; then
    for pid in $MCP_PIDS; do
        info "Killing pid $pid"
        kill "$pid" 2>/dev/null || true
    done
    sleep 2
    # Verify they're gone
    STILL=$(pgrep -f "src.mcp_server" 2>/dev/null || true)
    if [ -n "$STILL" ]; then
        warn "Some MCP pids still running: $STILL"
        for pid in $STILL; do
            kill -9 "$pid" 2>/dev/null || true
        done
        sleep 1
    fi
fi

# Step 4: signal the gateway
echo "  → Step 4: Signal gateway to restart"
if [ -n "$GATEWAY_PID" ]; then
    info "Sending SIGUSR1 to gateway pid $GATEWAY_PID"
    kill -USR1 "$GATEWAY_PID" 2>/dev/null || true
    sleep 3
fi

# Step 5: wait for respawn
echo "  → Step 5: Wait for MCP respawn"
for i in {1..30}; do
    if pgrep -f "src.mcp_server" > /dev/null 2>&1; then
        ok "MCP respawned after ${i}s"
        break
    fi
    sleep 1
done

if ! pgrep -f "src.mcp_server" > /dev/null 2>&1; then
    error "MCP did not respawn after 30s. Check the gateway."
    exit 1
fi

# Step 6: verify with a probe
echo "  → Step 6: Verify brain works"
PROBE=$("$PYTHON" -m src.cli openclaw call brain_stats '{}' 2>&1 || true)
if echo "$PROBE" | grep -q '"total"'; then
    ok "Brain responds to brain_stats"
else
    error "Brain did not respond. Output:"
    echo "$PROBE" | head -5
    exit 1
fi

echo ""
ok "Reconnect complete."
