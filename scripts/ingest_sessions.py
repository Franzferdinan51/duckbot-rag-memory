#!/usr/bin/env python3
"""ingest_sessions.py — distill agent session stores into the brain (Phase F).

Normalizes grok-local session histories and zcode-local rollouts and
runs them through the Hy-Memory writer pipeline (facts with Phase A
reconciliation, intentions, episodes). State-tracked: reruns only
process new/changed sessions.

Usage:
  python scripts/ingest_sessions.py [--sources grok,zcode] [--days 7]
      [--data-dir DIR] [--state FILE] [--dry-run]
"""
from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WRITER_MODEL = "hy-memory-30b-a3b-mlx"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="grok,zcode")
    ap.add_argument("--days", type=float, default=7)
    ap.add_argument("--model", default=WRITER_MODEL)
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--state", default=None,
                    help="ingest state file (default: ~/.duckbot-session-ingest.json)")
    ap.add_argument("--dry-run", action="store_true",
                    help="list candidate session files without ingesting")
    args = ap.parse_args()

    from src.session_ingest import discover_sessions, ingest_files

    files: list[tuple[str, Path]] = []
    for kind in [s.strip() for s in args.sources.split(",") if s.strip()]:
        found = discover_sessions(kind, days=args.days)
        files.extend((kind, p) for p in found)
        print(f"{kind}: {len(found)} session file(s) in the last {args.days}d")
    if args.dry_run:
        for _kind, p in files[:20]:
            print(f"  {p} ({p.stat().st_size // 1024} KiB)")
        return 0

    from src.memory import Memory

    state_path = Path(args.state) if args.state else (
        Path.home() / ".duckbot-session-ingest.json")
    lms = Path.home() / ".lmstudio" / "bin" / "lms"
    subprocess.run([str(lms), "load", args.model, "--context-length", "8192", "-y"],
                   capture_output=True, timeout=300)
    try:
        mem = Memory(persist_dir=args.data_dir) if args.data_dir else Memory()
        counts = asyncio.run(ingest_files(mem, files, model=args.model,
                                          state_path=state_path))
        print(f"INGESTED: files={counts['files']} skipped={counts['skipped']} "
              f"facts={counts['facts']} episodes={counts['episodes']} "
              f"intentions={counts['intentions']}")
        return 0
    finally:
        subprocess.run([str(lms), "unload", args.model],
                       capture_output=True, timeout=120)


if __name__ == "__main__":
    raise SystemExit(main())
