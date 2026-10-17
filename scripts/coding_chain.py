#!/usr/bin/env python3
"""coding_chain.py — build coding diff-chains for Duckets' repos (Phase E).

Walks each repo's git history, distills commits into decision records
with the pinned Hy-Memory MLX writer, and stores them chain-linked in
the bank. Idempotent — rerunning only stores new commits.

Usage:
  python scripts/coding_chain.py REPO... [--limit N] [--data-dir DIR]
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
    ap.add_argument("repos", nargs="+", help="repo paths")
    ap.add_argument("--model", default=WRITER_MODEL)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--data-dir", default=None)
    args = ap.parse_args()

    from src.coding import store_chain
    from src.memory import Memory

    lms = Path.home() / ".lmstudio" / "bin" / "lms"
    subprocess.run([str(lms), "load", args.model, "--context-length", "8192", "-y"],
                   capture_output=True, timeout=300)
    try:
        mem = Memory(persist_dir=args.data_dir) if args.data_dir else Memory()

        async def _run():
            for repo in args.repos:
                res = await store_chain(mem, repo, model=args.model,
                                        limit=args.limit)
                print(f"{res['repo']}: commits={res['commits']} "
                      f"stored={res['stored']} skipped={res['skipped']} "
                      f"fallback={res['fallback']}")

        asyncio.run(_run())
        return 0
    finally:
        subprocess.run([str(lms), "unload", args.model],
                       capture_output=True, timeout=120)


if __name__ == "__main__":
    raise SystemExit(main())
