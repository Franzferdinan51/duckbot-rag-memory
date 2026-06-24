#!/usr/bin/env bash
# scripts/cron.sh — nightly ingest + eval + commit.
# Designed to be invoked by cron every ~90 min between 22:00 and 10:00 EDT.
#
# Cron runs us with a minimal PATH and /bin/sh, so we cannot rely on `python`
# being on PATH. We resolve a working Python via:
#   1. PYTHON_BIN env var (manual override)
#   2. The venv at $REPO_ROOT/.venv/bin/python (preferred)
#   3. `which python3.11 python3.10 python3.9 python3` fallback chain
#   4. PY env var, if set
#
# Embedded embeddings are not used here — we require either:
#   - OPENAI_API_KEY in .env, OR
#   - LM Studio running on LMSTUDIO_URL (default http://127.0.0.1:1234)
# If neither is available the cron logs a warning and continues (the
# IngestRunner will skip embedding and the run is a no-op for retrieval,
# but the rest of the pipeline still works).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Export a sane PATH (cron usually gives /usr/bin:/bin only)
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/bin"

# 1. Load .env if present
if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

# 2. Resolve a working python
resolve_python() {
  if [[ -n "${PYTHON_BIN:-}" ]] && command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "$PYTHON_BIN"; return 0
  fi
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    echo "$REPO_ROOT/.venv/bin/python"; return 0
  fi
  if [[ -x "$REPO_ROOT/.venv/bin/python3" ]]; then
    echo "$REPO_ROOT/.venv/bin/python3"; return 0
  fi
  for cand in python3.11 python3.10 python3.9 python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
      command -v "$cand"; return 0
    fi
  done
  return 1
}

PY="$(resolve_python)" || {
  # No python found at all — try to bootstrap a venv using a system python
  log "WARN: no python found, attempting to bootstrap venv from system python3"
  for cand in python3.11 python3.10 python3.9 python3; do
    if command -v "$cand" >/dev/null 2>&1; then
      SYS_PY="$(command -v "$cand")"
      "$SYS_PY" -m venv "$REPO_ROOT/.venv" || { log "FATAL: venv create failed"; exit 127; }
      "$REPO_ROOT/.venv/bin/pip" install --quiet --upgrade pip
      "$REPO_ROOT/.venv/bin/pip" install --quiet -r "$REPO_ROOT/requirements.txt" || {
        log "FATAL: pip install failed"; exit 127
      }
      PY="$REPO_ROOT/.venv/bin/python"
      break
    fi
  done
  [[ -x "$PY" ]] || { echo "FATAL: no python on PATH" >&2; exit 127; }
}

# 3. Output / logging
OPENCLAW_MEMORY="${OPENCLAW_MEMORY:-$HOME/.openclaw/workspace/memory}"
OPENCLAW_WORKSPACE="${OPENCLAW_WORKSPACE:-$HOME/.openclaw/workspace}"
LOG_DIR="$REPO_ROOT/data/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/cron-$TS.log"

log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$LOG_FILE"; }

log "=== DuckBot RAG cron starting (python: $PY) ==="
log "Repo: $REPO_ROOT"
log "Source memory: $OPENCLAW_MEMORY"

# 4. Detect embedding availability
EMBEDDING_MODE="none"
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  EMBEDDING_MODE="openai"
elif [[ -n "${DUCKBOT_EMBEDDING:-}" && "${DUCKBOT_EMBEDDING}" == "local" ]]; then
  EMBEDDING_MODE="local"
elif [[ -n "${LMSTUDIO_URL:-}" ]] || curl -fsS --max-time 2 "${LMSTUDIO_URL:-http://127.0.0.1:1234}/v1/models" >/dev/null 2>&1; then
  EMBEDDING_MODE="lmstudio"
fi
log "Embedding mode: $EMBEDDING_MODE"

# 5. Helper to run a phase
run_phase() {
  local name="$1"; shift
  log "Phase: $name"
  if "$@" >> "$LOG_FILE" 2>&1; then
    log "$name OK"
    return 0
  else
    local rc=$?
    log "$name FAILED (exit $rc) — see $LOG_FILE"
    return $rc
  fi
}

INGEST_ARGS=(
  "$OPENCLAW_WORKSPACE/MEMORY.md"
  "$OPENCLAW_WORKSPACE/AGENTS.md"
  "$OPENCLAW_WORKSPACE/SOUL.md"
  "$OPENCLAW_WORKSPACE/IDENTITY.md"
  "$OPENCLAW_MEMORY"
)
# Opt-in: extra project paths via DUCKBOT_INGEST_EXTRA (colon-separated).
# The previous version hardcoded two paths that only existed on one
# operator's machine, making the script fragile for everyone else.
if [[ -n "${DUCKBOT_INGEST_EXTRA:-}" ]]; then
  IFS=':' read -r -a _extra_paths <<< "$DUCKBOT_INGEST_EXTRA"
  for p in "${_extra_paths[@]}"; do
    [[ -n "$p" ]] && INGEST_ARGS+=("$p")
  done
fi

# 6. Ingest (only if we have an embedding provider)
if [[ "$EMBEDDING_MODE" != "none" ]]; then
  run_phase "ingest" "$PY" -m src.cli ingest "${INGEST_ARGS[@]}" || true
  run_phase "consolidate" "$PY" -m src.cli consolidate 7 || true
else
  log "Phase: ingest — SKIPPED (no embedding provider; set OPENAI_API_KEY or DUCKBOT_EMBEDDING=local)"
fi

# 6b. Sync enhanced brain to agent context files (OpenClaw + Hermes).
# Writes MEMORY.md, USER.md, SOUL.md so agents that read these files
# on startup get a pre-loaded brain — not a blank slate.
if [[ "$EMBEDDING_MODE" != "none" ]]; then
  run_phase "brain-sync" "$PY" -m src.cli sync --target both || true
fi

# 7. Eval (only if benchmark exists AND we have embeddings)
BENCH="$REPO_ROOT/benchmarks/golden.jsonl"
if [[ -f "$BENCH" && "$EMBEDDING_MODE" != "none" ]]; then
  run_phase "eval" "$PY" -m src.cli eval "$BENCH" || true
else
  log "Phase: eval — SKIPPED (no benchmark or no embedding provider)"
fi

# 8. Stats snapshot
run_phase "stats" "$PY" -m src.cli stats || true

# 9. Commit progress (data/ is gitignored; logs + benchmarks are committed)
log "Phase: commit"
cd "$REPO_ROOT"
git add data/logs/ benchmarks/ src/ tests/ docs/ README.md AGENTS.md CHANGELOG.md 2>/dev/null || true
if git diff --cached --quiet; then
  log "No changes to commit"
else
  if git commit -m "cron: ingest + eval + consolidate at $TS" >> "$LOG_FILE" 2>&1; then
    log "Commit OK"
  else
    log "Commit FAILED (no remote? dirty tree?)"
  fi
fi

log "=== DuckBot RAG cron finished ==="
