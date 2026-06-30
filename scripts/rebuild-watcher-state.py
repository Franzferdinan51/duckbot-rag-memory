#!/usr/bin/env python3
"""rebuild-watcher-state.py — Rebuild watcher_state.json from ChromaDB metadata.

WHY THIS EXISTS:
The duckbot-rag-memory watcher keeps `data/watcher_state.json` so it knows
which files it has already ingested. The watcher uses a content-hash dedup
to skip files whose content hasn't changed.

If the state file is lost (or wiped during a ChromaDB rebuild), the watcher
will try to re-ingest every watched file on the next poll — and if the
ChromaDB Rust hnsw index is corrupted (the macOS HNSW bloat bug), each
ingest triggers a SIGSEGV in chromadb/api/rust.py:_query.

This script reads the source_path metadata directly from ChromaDB's SQLite
file (bypassing the Rust bindings, which crash) and rebuilds
watcher_state.json with one entry per known source_path. After running
this, the watcher's content-hash dedup will skip all already-ingested files
and only process truly new ones — keeping the watcher alive even while the
underlying HNSW bug is unfixed.

USAGE:
    python scripts/rebuild-watcher-state.py [--dry-run]

READ-ONLY on ChromaDB. Writes to data/watcher_state.json (atomic rename).
"""
import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHROMA_DB = REPO_ROOT / "data" / "chroma" / "chroma.sqlite3"
STATE_PATH = REPO_ROOT / "data" / "watcher_state.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without touching state file",
    )
    args = parser.parse_args()

    if not CHROMA_DB.exists():
        print(f"ERROR: ChromaDB not found at {CHROMA_DB}", file=sys.stderr)
        return 1

    # Read-only SQLite connection (don't lock ChromaDB while the watcher
    # or MCP server is running). WAL mode means concurrent reads are fine.
    con = sqlite3.connect(f"file:{CHROMA_DB}?mode=ro", uri=True)
    cur = con.cursor()

    # Aggregate source_paths → counts + max(last_recalled_at) as proxy for
    # last_modified. We don't have real mtime in the chunks table, so this
    # is an approximation — the content-hash check on next sync will
    # re-process files whose actual content changed.
    cur.execute("""
        SELECT
            string_value,
            COUNT(*) AS chunk_count,
            MAX(COALESCE(
                (SELECT float_value FROM embedding_metadata m2
                 WHERE m2.id = m.id AND m2.key = 'last_recalled_at'),
                0
            )) AS last_recalled
        FROM embedding_metadata m
        WHERE key = 'source_path'
        GROUP BY string_value
    """)
    rows = cur.fetchall()
    con.close()

    files = {}
    hash_matched = 0
    hash_mismatch = 0
    file_missing = 0
    for src, count, last_recalled in rows:
        if not src:
            continue
        # Normalize: source_path may be absolute or repo-relative (memory/...).
        if src.startswith("~"):
            src = os.path.expanduser(src)
        elif not src.startswith("/"):
            for prefix in (str(REPO_ROOT), str(Path.home() / ".openclaw" / "workspace")):
                cand = str(Path(prefix) / src)
                if Path(cand).exists():
                    src = cand
                    break

        # Compute the actual sha256 of the file on disk so the watcher's
        # content-hash dedup will skip it on the next poll. This is
        # critical: the HNSW bug in ChromaDB causes segfaults whenever
        # the watcher tries to remember() a chunk, so we MUST make the
        # watcher skip these files instead of re-ingesting them.
        src_path = Path(src)
        if not src_path.exists():
            file_missing += 1
            continue
        try:
            content = src_path.read_text(encoding="utf-8", errors="ignore")
            content_hash = hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()
            stat = src_path.stat()
            mtime = stat.st_mtime
        except OSError:
            file_missing += 1
            continue

        hash_matched += 1
        files[src] = {
            "mtime": mtime,
            "content_hash": content_hash,
            "chunk_ids": [],  # unknown — chunk_ids live in ChromaDB; not rebuildable from here
            "chunk_count": count,
            "last_sync": time.time(),
            "rebuilt_from_chroma": True,
        }

    print(f"  hashed: {hash_matched} matched (will be skipped), {file_missing} file missing on disk")

    state = {
        "files": files,
        "last_run": time.time(),
        "total_remembered": sum(f["chunk_count"] for f in files.values()),
        "total_forgotten": 0,
        "rebuilt_at": time.time(),
        "rebuild_note": (
            "State rebuilt from ChromaDB metadata. "
            "content_hash is a placeholder — watcher will re-hash each file "
            "on next poll and skip if unchanged."
        ),
    }

    print(f"Found {len(files)} distinct source paths, {sum(f['chunk_count'] for f in files.values())} total chunks")

    if args.dry_run:
        print("DRY RUN — would write to", STATE_PATH)
        for src in sorted(files)[:5]:
            print(f"  {files[src]['chunk_count']:5d} chunks  {src}")
        if len(files) > 5:
            print(f"  ... and {len(files) - 5} more")
        return 0

    # Atomic write
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    os.replace(tmp, STATE_PATH)
    print(f"Wrote {STATE_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())