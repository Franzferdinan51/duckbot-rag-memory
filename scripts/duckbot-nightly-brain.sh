#!/usr/bin/env bash
# duckbot-nightly-brain.sh — nightly DuckBot brain loop (Robin, systemd timer).
#
# Canonical copy of ~/rag-jobs/duckbot-nightly-brain.sh on Robin.
# Keep the two in sync; the Robin copy is what the timer runs.
#
# 1. Ensure the embedder is loaded (reflect promotion + sync need it).
# 2. Nightly consolidation pass with the light writer (load-on-dispatch;
#    chat model untouched). Writer: lfm3-omni-2b (Hy-Memory 30B retired
#    2026-11-26 — too heavy, and its load failed nightly on Robin).
# 3. Sync brain -> Hermes files (SOUL.md / MEMORY.md / USER.md).
#
# stdout is posted verbatim to the delivery target, so keep it one line.
#
# 2026-11-26 fixes vs the previous version:
#   - The writer pass exit code was captured AFTER writing the log file,
#     so RC was always printf's 0 and a failed pass could never reach the
#     FAILED branch (18 straight nights of masked Hy load failures).
#     RC is now captured immediately after the pass runs.
#   - The D1 health check pointed at ~/.hermes/tools/d1/d1.py, a Mac path
#     that does not exist on Robin (brain=skipped-no-d1 every night). It
#     now probes the LM Studio API (via the tunnel) for d1-2b.
#   - RAM free used macOS `memory_pressure`; now uses Linux `free`.
set -uo pipefail
REPO="$HOME/duckbot-rag-memory"
cd "$REPO" || { echo "🧠 DuckBot brain nightly: repo missing at $REPO"; exit 0; }
unset PYTHONPATH

# 1. Embedder must be loaded for the promotion + sync embedding paths.
if ! lms ps --json 2>/dev/null | grep -q 'nomic-embed'; then
  lms load text-embedding-nomic-embed-text-v1.5 -y >/dev/null 2>&1 || true
  sleep 5
fi

# 2. Nightly consolidation (30d episodic lookback).
OUT=$( "$REPO/.venv/bin/python" scripts/nightly_hy_pass.py --days 30 2>&1 )
RC=$?
mkdir -p "$HOME/rag-jobs/logs"; printf '%s\n' "$OUT" > "$HOME/rag-jobs/logs/nightly-$(date +%F).log"
if [ $RC -ne 0 ]; then
  echo "🧠 DuckBot brain nightly: writer pass FAILED (rc=$RC) — $(echo "$OUT" | grep -iE 'error|failed' | tail -1 | head -c 150)"
  exit 0
fi

# 2b. Decision model — is the D1 gate model reachable for tonight's runs?
if curl -s --max-time 10 http://127.0.0.1:1234/v1/models 2>/dev/null | grep -q 'd1-2b'; then
  DEC_STATUS="d1-2b-ok"
else
  DEC_STATUS="d1-unreachable"
fi

# 3. Sync brain -> Hermes context files.
if "$REPO/.venv/bin/python" -m src.cli sync --target hermes >/dev/null 2>&1; then
  SYNCED="synced"
else
  SYNCED="sync-failed"
fi

# 4. Environment state (standing directive: always know what the machine can run).
LOADED=$(lms ps --json 2>/dev/null | grep -o '"type"' | wc -l | tr -d ' ')
FREEPCT=$(free -m 2>/dev/null | awk '/^Mem:/{if (NF>=7) printf "%d%%", $7*100/$2; else printf "%d%%", $4*100/$2}')

echo "🧠 DuckBot brain nightly: $(echo "$OUT" | tail -1 | head -c 90) | SOUL/MEMORY/USER $SYNCED | brain=$DEC_STATUS | env: $LOADED models loaded, RAM ${FREEPCT:-?} free"

# Cleanup: unload the embedder we self-loaded (frees ~84 MB; next run self-heals).
lms unload text-embedding-nomic-embed-text-v1.5 >/dev/null 2>&1 || true
