#!/usr/bin/env python3
"""ingest_sessions.py — distill agent session stores into the brain (Phase F).

Normalizes grok-local session histories and zcode-local rollouts and
runs them through the Hy-Memory writer pipeline (facts with Phase A
reconciliation, intentions, episodes). State-tracked: reruns only
process new/changed sessions.

Usage:
  python scripts/ingest_sessions.py --sources grok,zcode [--days 7]
      [--data-dir DIR] [--state FILE] [--dry-run]
  python scripts/ingest_sessions.py --files EXPORT.jsonl [MORE...]
      [--dry-run]

Importing is a MANUAL feature: the brain never ingests on its own.
Nothing runs unless the owner names a source or hands over files.
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
    ap.add_argument("--sources", default="",
                    help="comma list: grok,zcode (named explicitly - no default)")
    ap.add_argument("--files", nargs="*", default=[],
                    help="exported session files (.jsonl/.md/.txt) from any agent")
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
    for f in args.files:
        p = Path(f).expanduser()
        if p.is_dir():
            hits = sorted(q for q in p.rglob("*")
                          if q.suffix.lower() in (".jsonl", ".md", ".txt"))
            files.extend(("files", q) for q in hits)
            print(f"files: {len(hits)} export file(s) under {p}")
        elif p.exists():
            files.append(("files", p))
            print(f"files: 1 export file ({p})")
        else:
            print(f"files: NOT FOUND: {p}")
    if not files and not args.dry_run:
        print("nothing to import: name --sources grok,zcode and/or --files ...")
        return 2
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
