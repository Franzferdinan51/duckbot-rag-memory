#!/bin/bash
# Restore duckbot-rag-memory from a backup archive.
#
# Usage:
#   scripts/restore-brain.sh                          # restore latest backup in data/backups/
#   scripts/restore-brain.sh <path-to-backup.tar.gz>  # restore from a specific archive
#   scripts/restore-brain.sh --dry-run                # verify archive integrity without writing
#
# What this does:
#   1. Stops the watcher (launchctl unload) so it doesn't fight the restore.
#   2. Extracts the archive into a sibling directory (e.g. duckbot-rag-memory-restored/)
#      so we NEVER overwrite the live brain by accident.
#   3. Runs `src.cli import --in-path data/brain_export.md` to re-ingest chunks
#      into ChromaDB if the live HNSW is empty or corrupt.
#   4. Restarts the watcher (launchctl load).
#
# Safety:
#   - Refuses to extract into $BRAIN_DIR itself.
#   - Refuses to import into a non-empty ChromaDB unless --force is passed.
#   - Prints the diff (size + chunk count) before applying anything destructive.
#
# Use this any time you need to validate a backup, or after a wipe to repopulate.

set -eo pipefail

BRAIN_DIR="/Users/duckets/Desktop/duckbot-rag-memory"
PYTHON="$BRAIN_DIR/.venv/bin/python"
PLIST="$HOME/Library/LaunchAgents/com.duckbot.memory-watcher.plist"
BACKUPS_DIR="$BRAIN_DIR/data/backups"
DRY_RUN=0
ARCHIVE=""
FORCE=0

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        --force) FORCE=1 ;;
        --help|-h)
            head -25 "$0" | tail -22
            exit 0
            ;;
        *) ARCHIVE="$arg" ;;
    esac
done

# Pick latest backup if none specified.
if [ -z "$ARCHIVE" ]; then
    ARCHIVE="$(ls -t "$BACKUPS_DIR"/brain-backup-*.tar.gz 2>/dev/null | head -1 || true)"
    if [ -z "$ARCHIVE" ]; then
        echo "[restore] no backups found under $BACKUPS_DIR" >&2
        exit 1
    fi
fi

if [ ! -f "$ARCHIVE" ]; then
    echo "[restore] archive not found: $ARCHIVE" >&2
    exit 1
fi

echo "[restore] archive: $ARCHIVE"
SIZE=$(du -h "$ARCHIVE" | awk '{print $1}')
ENTRIES=$(tar -tzf "$ARCHIVE" | wc -l | tr -d ' ')
echo "[restore] size: $SIZE, entries: $ENTRIES"

# Extract into a sibling directory so we never overwrite the live brain.
RESTORE_DIR="$BRAIN_DIR-restored-$(date +%Y%m%d_%H%M%S)"
echo "[restore] will extract into: $RESTORE_DIR"

if [ "$DRY_RUN" = "1" ]; then
    echo "[restore] --dry-run: skipping extract + import"
    echo "[restore] archive contents (first 20):"
    tar -tzf "$ARCHIVE" | head -20
    echo "[restore] manifest preview:"
    tar -xzf "$ARCHIVE" MANIFEST.md -O 2>/dev/null | head -30
    exit 0
fi

mkdir -p "$RESTORE_DIR"
tar -xzf "$ARCHIVE" -C "$RESTORE_DIR"
echo "[restore] ✓ extracted to $RESTORE_DIR"

# Validate the restored brain_export.md is parseable.
EXPORT="$RESTORE_DIR/data/brain_export.md"
if [ ! -f "$EXPORT" ]; then
    echo "[restore] FATAL: archive missing data/brain_export.md" >&2
    exit 1
fi
LINES=$(wc -l < "$EXPORT")
echo "[restore] brain_export.md: $LINES lines"

# Refuse to import into the live brain without --force, since import APPENDS.
LIVE_CHROMA="$BRAIN_DIR/data/chroma"
if [ -d "$LIVE_CHROMA" ]; then
    EXISTING=$(find "$LIVE_CHROMA" -name '*.sqlite3' -exec ls -la {} \; | wc -l | tr -d ' ')
    if [ "$EXISTING" -gt 0 ] && [ "$FORCE" != "1" ]; then
        echo "[restore] live chroma has $EXISTING SQLite db(s); refusing to import without --force"
        echo "[restore] to re-import anyway (will APPEND to existing chunks):"
        echo "         $0 $ARCHIVE --force"
        exit 2
    fi
fi

# Stop the watcher so it doesn't fight the restore.
launchctl unload "$PLIST" 2>/dev/null || true
echo "[restore] watcher unloaded"

cd "$BRAIN_DIR"
echo "[restore] importing brain_export.md into live chroma..."
"$PYTHON" -m src.cli import --in-path "$EXPORT" 2>&1 | tail -10

# Restart the watcher.
launchctl load -w "$PLIST" 2>/dev/null || true
echo "[restore] watcher reloaded"

echo
echo "[restore] ✓ done. extracted at: $RESTORE_DIR"
echo "[restore] inspect with:  ls $RESTORE_DIR"
echo "[restore] to overwrite live brain with the restored copy:"
echo "         rsync -a --delete $RESTORE_DIR/data/ $BRAIN_DIR/data/"
echo "         (then run src.cli import --in-path data/brain_export.md)"