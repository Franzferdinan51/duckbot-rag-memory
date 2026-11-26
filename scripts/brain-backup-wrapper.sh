#!/usr/bin/env bash
# brain-backup-wrapper.sh — weekly brain backup (Robin, systemd timer).
#
# Canonical copy of ~/rag-jobs/brain-backup-wrapper.sh on Robin.
# Runs scripts/backup-brain.sh (full state archive: export + chroma +
# sqlite dbs) into data/backups/ and keeps only the newest 4 archives.
# Before v0.18.0 the backup script existed but nothing ever ran it —
# the brain had no scheduled backup at all.
set -uo pipefail
REPO="$HOME/duckbot-rag-memory"
cd "$REPO" || { echo "brain backup: repo missing"; exit 0; }

OUT=$(bash scripts/backup-brain.sh 2>&1)
RC=$?
if [ $RC -ne 0 ]; then
  echo "brain backup: FAILED (rc=$RC) — $(echo "$OUT" | tail -1 | head -c 200)"
  exit 1
fi
echo "brain backup: $(echo "$OUT" | tail -2 | tr '\n' ' ')"

# Prune to the newest 4 archives.
ls -1t data/backups/brain-backup-*.tar.gz 2>/dev/null | tail -n +5 | while read -r old; do
  rm -f "$old"
  echo "brain backup: pruned $old"
done
ls -lh data/backups/ 2>/dev/null | tail -3
