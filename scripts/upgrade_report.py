#!/usr/bin/env python3
"""upgrade_report.py — print the Hy-Memory upgrade acceptance report.

Layer census + graph stats for a bank (--data-dir, default the live
bank), plus the golden recall benchmark unless --no-golden. All reads
are recall/metadata only; nothing is written to the bank.

Usage: python scripts/upgrade_report.py [--data-dir DIR] [--no-golden]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--golden", default="benchmarks/golden.jsonl")
    ap.add_argument("--no-golden", action="store_true")
    args = ap.parse_args()

    from src.memory import Memory
    from src.upgrade_report import build_report

    mem = Memory(persist_dir=args.data_dir) if args.data_dir else Memory()
    benchmark = None if args.no_golden else Path(args.golden)
    report = asyncio.run(build_report(mem, benchmark=benchmark))
    print(json.dumps(report, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
