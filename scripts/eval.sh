#!/usr/bin/env bash
# scripts/eval.sh — manual eval run with formatted output.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"
[[ -f .env ]] && set -a && source .env && set +a
BENCH="${1:-$REPO_ROOT/benchmarks/golden.jsonl}"
if [[ ! -f "$BENCH" ]]; then
  echo "No benchmark at $BENCH" >&2
  exit 1
fi
python -m src.cli eval "$BENCH" | python -m json.tool