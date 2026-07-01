#!/usr/bin/env bash
# cleanup-test-artifacts.sh — Remove test artifacts from the brain's live state.
#
# Usage:
#   ~/Desktop/duckbot-rag-memory/scripts/cleanup-test-artifacts.sh
#   ~/Desktop/duckbot-rag-memory/scripts/cleanup-test-artifacts.sh --dry-run
#
# What it removes:
#   - skills/e2e-smoke-skill/, skills/looping-fix-test/, etc. (any slug
#     matching *_test or smoke_* or e2e_*)
#   - Memory blocks named e2e_*, auth_*, smoke_*, test_*
#   - Test chunks matching patterns (use brain_forget_by_query)
#
# Always dry-runs first to show what will be deleted.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

DRY_RUN=true
for arg in "$@"; do
    case "$arg" in
        --no-dry-run) DRY_RUN=false ;;
        --dry-run) DRY_RUN=true ;;
        --help|-h)
            echo "Usage: cleanup-test-artifacts.sh [--no-dry-run]"
            echo "Default is dry-run. Pass --no-dry-run to actually delete."
            exit 0
            ;;
    esac
done

if [[ -x .venv/bin/python ]]; then
    PYTHON=".venv/bin/python"
elif [[ -x .venv/Scripts/python.exe ]]; then
    PYTHON=".venv/Scripts/python.exe"
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

echo -e "${BOLD}🧠  DuckBot brain — cleanup test artifacts${RESET}"
echo ""

if $DRY_RUN; then
    warn "DRY RUN. Pass --no-dry-run to actually delete."
    echo ""
fi

# 1. Skills dir
echo -e "${CYAN}Skills directory ($REPO_ROOT/skills/):${RESET}"
for d in "$REPO_ROOT"/skills/; do
    for sub in "$d"/*/; do
        [ -d "$sub" ] || continue
        name=$(basename "$sub")
        if [[ "$name" == e2e-* || "$name" == *-test || "$name" == smoke* || "$name" == looping-* || "$name" == auth_* || "$name" == test_* ]]; then
            if $DRY_RUN; then
                warn "would remove: $sub"
            else
                rm -rf "$sub"
                ok "removed: $name"
            fi
        fi
    done
done

# 2. Test blocks via Python
echo ""
echo -e "${CYAN}Test memory blocks:${RESET}"
"$PYTHON" -c "
import sys
sys.path.insert(0, '$REPO_ROOT')
import sqlite3
from pathlib import Path
db = Path('$REPO_ROOT/data/blocks.db')
if not db.exists():
    print(f'  no blocks.db at {db}')
    sys.exit(0)
conn = sqlite3.connect(str(db))
cur = conn.cursor()
patterns = ('e2e_%', 'auth_%', 'smoke_%', 'test_%', 'looping_%', 'junk_%')
all_rows = []
for pat in patterns:
    rows = cur.execute('SELECT name FROM blocks WHERE name LIKE ?', (pat,)).fetchall()
    for r in rows:
        all_rows.append(r[0])
if not all_rows:
    print('  no test blocks found')
else:
    for name in all_rows:
        if $DRY_RUN:
            print(f'  would remove block: {name}')
        else:
            cur.execute('DELETE FROM blocks WHERE name = ?', (name,))
            cur.execute('DELETE FROM block_history WHERE name = ?', (name,))
            conn.commit()
            print(f'  removed block: {name}')
conn.close()
"

# 3. Test chunks
echo ""
echo -e "${CYAN}Test chunks (via brain_forget_by_query):${RESET}"
for query in "looping-fix-test" "e2e smoke" "test_rerank_e2e" "auth_test_block" "Project Zorgo"; do
    if $DRY_RUN; then
        info "would forget chunks matching: $query"
    else
        "$PYTHON" -m src.cli openclaw call brain_forget_by_query "{\"query\": \"$query\", \"k\": 20}" > /dev/null 2>&1 || true
        ok "forgot chunks matching: $query"
    fi
done

echo ""
if $DRY_RUN; then
    warn "DRY RUN complete. Run with --no-dry-run to actually delete."
else
    ok "Cleanup complete."
fi
