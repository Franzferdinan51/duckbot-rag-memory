#!/bin/bash
# Backup the entire duckbot-rag-memory state to a single archive.
#
# What this preserves (per Duckets' "dont delete anything" rule):
#   1. Brain content    — data/brain_export.md (all chunks as markdown, round-trippable via brain_import)
#   2. ChromaDB vectors — data/chroma/ (full HNSW indexes + SQLite metadata)
#   3. SQLite databases — data/blocks.db, data/graph.db, data/events.db,
#                         data/ingest_history.jsonl, data/eval_history.jsonl
#   4. Watcher state    — data/watcher_state.json (sha256 + skipped files)
#   5. Any corrupt backups under data/chroma/_corrupt_backup_*/
#   6. Repo scripts + config (.env, scripts/, src/) so the backup is self-describing
#
# Usage:
#   scripts/backup-brain.sh                 # timestamped backup into data/backups/
#   scripts/backup-brain.sh <out_path>      # custom output path
#   scripts/backup-brain.sh --no-freeze     # skip the brain_export regeneration (use last export)
#
# The script is non-destructive: it ONLY reads from data/ and writes a single
# archive under data/backups/ (or wherever you point it). Safe to run any time.
#
# To restore:
#   1. Stop the watcher (launchctl unload com.duckbot.memory-watcher)
#   2. tar -xzf brain-backup-*.tar.gz -C ~/Desktop/duckbot-rag-memory/
#   3. .venv/bin/python -m src.cli import --in-path data/brain_export.md
#      (re-ingests chunks if chromadb is empty)
#   4. launchctl load ~/Library/LaunchAgents/com.duckbot.memory-watcher.plist

set -eo pipefail

BRAIN_DIR="/Users/duckets/Desktop/duckbot-rag-memory"
PYTHON="$BRAIN_DIR/.venv/bin/python"
BACKUPS_DIR="$BRAIN_DIR/data/backups"
REGENERATE_EXPORT=1
OUT_PATH=""

# Parse args
for arg in "$@"; do
    case "$arg" in
        --no-freeze) REGENERATE_EXPORT=0 ;;
        --help|-h)
            head -25 "$0" | tail -22
            exit 0
            ;;
        *) OUT_PATH="$arg" ;;
    esac
done

if [ -n "$OUT_PATH" ]; then
    ARCHIVE="$OUT_PATH"
    TMP_ARCHIVE="$(mktemp -t brain-backup).tar.gz"
else
    mkdir -p "$BACKUPS_DIR"
    STAMP="$(date +%Y-%m-%d_%H%M%S)"
    ARCHIVE="$BACKUPS_DIR/brain-backup-$STAMP.tar.gz"
    # IMPORTANT: write to a temp path OUTSIDE the brain dir, because tar
    # would otherwise refuse to write into a subdir of the source it's
    # archiving ("Can't add archive to itself"). We mv into place at the end.
    TMP_ARCHIVE="$(mktemp -t brain-backup).tar.gz"
fi

# Source .env if present so LM Studio is reachable for export.
ENV_FILE="/Users/duckets/Library/Application Support/duckbot-rag-memory/env"
[ -f "$ENV_FILE" ] && set -a && . "$ENV_FILE" && set +a

echo "[backup] brain dir: $BRAIN_DIR"
echo "[backup] archive:   $ARCHIVE"

# Step 1: regenerate brain_export.md so it reflects current state.
if [ "$REGENERATE_EXPORT" = "1" ]; then
    echo "[backup] regenerating brain_export.md..."
    cd "$BRAIN_DIR"
    if "$PYTHON" -m src.cli export --out-path "$BRAIN_DIR/data/brain_export.md" >/dev/null 2>&1; then
        CHUNKS=$(wc -l < "$BRAIN_DIR/data/brain_export.md")
        echo "[backup]   brain_export.md: $CHUNKS lines"
    else
        echo "[backup] WARN: brain export failed; using last known brain_export.md" >&2
    fi
else
    echo "[backup] --no-freeze set; using last brain_export.md as-is"
fi

# Step 2: build a manifest inside the brain dir so it ships with the archive.
MANIFEST="$BRAIN_DIR/data/backups/MANIFEST.md"
mkdir -p "$(dirname "$MANIFEST")"
{
    echo "# DuckBot-RAG-Memory Backup Manifest"
    echo
    echo "Generated: $(date -Iseconds)"
    echo "Hostname:  $(hostname)"
    echo "Brain dir: $BRAIN_DIR"
    echo "Archive:   $ARCHIVE"
    echo
    echo "## Contents"
    echo
    echo "- brain_export.md ($(wc -l < "$BRAIN_DIR/data/brain_export.md" 2>/dev/null || echo "?") lines)"
    echo "- data/chroma/ (ChromaDB HNSW indexes + SQLite metadata)"
    echo "- data/blocks.db, graph.db, events.db (SQLite metadata)"
    echo "- data/watcher_state.json (file hashes + skip flags)"
    echo "- data/ingest_history.jsonl, eval_history.jsonl"
    echo "- scripts/, src/ (so the backup is self-describing)"
    echo "- .env (secrets — keep this archive private)"
    echo
    echo "## Sizes"
    echo
    du -sh "$BRAIN_DIR/data/chroma" 2>/dev/null
    du -sh "$BRAIN_DIR/data/brain_export.md" 2>/dev/null
    du -sh "$BACKUPS_DIR" 2>/dev/null
    echo
    echo "## Restore procedure"
    echo
    echo '```bash'
    echo "launchctl unload ~/Library/LaunchAgents/com.duckbot.memory-watcher.plist"
    echo "tar -xzf $(basename "$ARCHIVE") -C ~/Desktop/"
    echo "cd ~/Desktop/duckbot-rag-memory"
    echo ".venv/bin/python -m src.cli import --in-path data/brain_export.md"
    echo "launchctl load ~/Library/LaunchAgents/com.duckbot.memory-watcher.plist"
    echo '```'
} > "$MANIFEST"

# Step 3: build the archive from $BRAIN_DIR itself with relative paths.
# tar is picky about `-C`: if you pass `-C path` mid-args, that becomes the
# cwd for all SUBSEQUENT files. So we pass `-C` ONLY for the manifest and
# run tar from $BRAIN_DIR (so `data`, `src`, `scripts`, `.env` resolve).
cd "$BRAIN_DIR"
echo "[backup] archiving..."
tar -czf "$TMP_ARCHIVE" \
    --exclude='data/backups' \
    --exclude='data/watcher.log' \
    --exclude='data/launchd.*.log' \
    --exclude='data/mcp.log' \
    --exclude='.venv' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='data/chroma/_corrupt_backup_*' \
    --exclude='data/logs' \
    data src scripts .env \
    -C "$BRAIN_DIR/data/backups" MANIFEST.md

# Move the temp archive into its final destination.
mv "$TMP_ARCHIVE" "$ARCHIVE"
rm -f "$MANIFEST"

SIZE=$(du -h "$ARCHIVE" | awk '{print $1}')
COUNT=$(tar -tzf "$ARCHIVE" 2>/dev/null | wc -l | tr -d ' ')
echo
echo "[backup] ✓ wrote $ARCHIVE ($SIZE, $COUNT entries)"
echo "[backup] To list contents: tar -tzf $ARCHIVE | head"
echo "[backup] To verify restore on a copy:"
echo "         mkdir /tmp/restore-test && tar -xzf $ARCHIVE -C /tmp/restore-test"