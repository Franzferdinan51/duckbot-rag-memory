#!/usr/bin/env bash
# status.sh — One-line health check for the DuckBot brain.
#
# Usage:
#   ~/Desktop/duckbot-rag-memory/scripts/status.sh
#   ~/Desktop/duckbot-rag-memory/scripts/status.sh --json
#   ~/Desktop/duckbot-rag-memory/scripts/status.sh --verbose
#
# Wraps `python -m src.cli doctor` + `python -m src.cli dashboard` +
# watcher/cron checks. Returns non-zero on any failure.
#
# Exit codes:
#   0 = all green
#   1 = some checks failed (printed in yellow)
#   2 = critical checks failed (brain can't be used)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

JSON=false
VERBOSE=false
SHOW_DASHBOARD=false
for arg in "$@"; do
    case "$arg" in
        --json) JSON=true ;;
        --verbose|-v) VERBOSE=true ;;
        --dashboard) SHOW_DASHBOARD=true ;;
        --help|-h)
            echo "Usage: status.sh [--json] [--verbose] [--dashboard]"
            echo ""
            echo "Health check for the DuckBot brain."
            echo "  --json       output JSON instead of human-readable"
            echo "  --verbose    also show brain stats and dashboard"
            echo "  --dashboard  show only the dashboard (skip doctor)"
            exit 0
            ;;
    esac
done

if [[ -x .venv/bin/python ]]; then
    PYTHON=".venv/bin/python"
elif [[ -x .venv/Scripts/python.exe ]]; then
    PYTHON=".venv/Scripts/python.exe"
else
    echo "✗ CRITICAL: no .venv python at $REPO_ROOT/.venv"
    exit 2
fi

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

EXIT_CODE=0

echo -e "${BOLD}🧠  DuckBot brain — health check${RESET}"
echo ""

# Doctor
echo -e "${CYAN}Doctor:${RESET}"
if $JSON; then
    DOCTOR_OUT=$("$PYTHON" -m src.cli doctor --json 2>&1) || true
    echo "$DOCTOR_OUT"
    if echo "$DOCTOR_OUT" | grep -q '"ok": false' 2>/dev/null; then
        EXIT_CODE=1
    fi
else
    if ! "$PYTHON" -m src.cli doctor 2>&1; then
        EXIT_CODE=1
    fi
fi

# Watcher
echo ""
echo -e "${CYAN}Watcher daemon:${RESET}"
WATCHER_PID=""
# Check launchd first (preferred on macOS)
if command -v launchctl > /dev/null 2>&1; then
    if launchctl list 2>/dev/null | grep -q "com.duckbot.memory-watcher"; then
        LAUNCHD_PID=$(launchctl list 2>/dev/null | awk '/com.duckbot.memory-watcher/ {print $1}' | head -1)
        if [ -n "$LAUNCHD_PID" ] && [ "$LAUNCHD_PID" != "-" ]; then
            echo -e "  ${GREEN}✓${RESET} launchd-managed (pid $LAUNCHD_PID)"
            WATCHER_PID="$LAUNCHD_PID"
        else
            echo -e "  ${YELLOW}!${RESET} launchd plist loaded but not running (last status: $LAUNCHD_PID)"
        fi
    fi
fi
# Fallback: pgrep (for non-launchd starts)
if [ -z "$WATCHER_PID" ] && pgrep -f "src.watcher" > /dev/null 2>&1; then
    WATCHER_PID=$(pgrep -f "src.watcher" | head -1)
    echo -e "  ${GREEN}✓${RESET} running (pid $WATCHER_PID)"
fi
if [ -z "$WATCHER_PID" ]; then
    echo -e "  ${YELLOW}!${RESET} not running (start with scripts/run-watcher.sh)"
    EXIT_CODE=$(( EXIT_CODE > 0 ? EXIT_CODE : 1 ))
fi

# Cron
echo ""
echo -e "${CYAN}Cron schedule:${RESET}"
if command -v crontab > /dev/null 2>&1; then
    CRON_OUT=$(crontab -l 2>&1 || true)
    if echo "$CRON_OUT" | grep -q "scripts/cron.sh"; then
        echo -e "  ${GREEN}✓${RESET} scripts/cron.sh scheduled"
    else
        echo -e "  ${YELLOW}!${RESET} scripts/cron.sh NOT in crontab"
        echo -e "    Run: ${BOLD}crontab -e${RESET} and add:"
        echo -e "    ${BOLD}0 2 * * * $REPO_ROOT/scripts/cron.sh${RESET}"
        EXIT_CODE=$(( EXIT_CODE > 0 ? EXIT_CODE : 1 ))
    fi
else
    echo -e "  ${YELLOW}!${RESET} crontab not available (Windows or sandboxed env)"
fi

# Dashboard (optional)
if $VERBOSE || $SHOW_DASHBOARD; then
    echo ""
    echo -e "${CYAN}Dashboard:${RESET}"
    "$PYTHON" -m src.cli dashboard 2>&1 || EXIT_CODE=$(( EXIT_CODE > 0 ? EXIT_CODE : 1 ))
fi

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ All green. Brain is healthy.${RESET}"
else
    echo -e "${YELLOW}! Brain usable but some optional checks failed.${RESET}"
fi

exit $EXIT_CODE
