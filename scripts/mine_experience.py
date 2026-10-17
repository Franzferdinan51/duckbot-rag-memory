#!/usr/bin/env python3
"""mine_experience.py — backfill experience episodes from session texts.

Reads markdown/text files (session summaries, brain files, agent logs),
mines reusable episodes {task, context, steps, outcome} with the pinned
Hy-Memory MLX writer, and stores them as layer="experience" chunks
(Phase D). Load-on-dispatch: the writer is loaded for the run and
unloaded after, same discipline as consolidate_job.py.

Usage:
  python scripts/mine_experience.py FILE... [--data-dir DIR] [--dry-run]
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
    ap.add_argument("paths", nargs="+", help="text/markdown files to mine")
    ap.add_argument("--model", default=WRITER_MODEL)
    ap.add_argument("--data-dir", default=None,
                    help="bank persist dir (default: the live bank)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print mined episodes without storing")
    args = ap.parse_args()

    from src.experience import mine_episodes
    from src.memory import Memory

    lms = Path.home() / ".lmstudio" / "bin" / "lms"
    subprocess.run([str(lms), "load", args.model, "--context-length", "8192", "-y"],
                   capture_output=True, timeout=300)
    try:
        mem = None if args.dry_run else (
            Memory(persist_dir=args.data_dir) if args.data_dir else Memory())

        async def _run() -> int:
            total = 0
            for p in args.paths:
                text = Path(p).read_text(errors="replace")
                if args.dry_run:
                    from src.hy_memory import extract_episodes
                    eps = extract_episodes(text, model=args.model)
                    for ep in eps:
                        print(f"  [{ep['outcome']}] {ep['task']}")
                    total += len(eps)
                else:
                    eps = await mine_episodes(mem, text, model=args.model,
                                              source_path=str(p))
                    total += len(eps)
                    print(f"{p}: {len(eps)} episode(s)")
            return total

        total = asyncio.run(_run())
        print(f"{'MINED (dry run)' if args.dry_run else 'STORED'}: {total} episode(s)")
        return 0
    finally:
        subprocess.run([str(lms), "unload", args.model],
                       capture_output=True, timeout=120)


if __name__ == "__main__":
    raise SystemExit(main())
