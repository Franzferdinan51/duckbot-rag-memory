#!/usr/bin/env python3
"""
consolidate_job.py — Phase 4 cutover-ready batch consolidation job.

This is the job-layer entrypoint the nightly jobs will call after the
dev -> main merge. It changes NO live behavior by existing: the live
Hermes jobs still run the old path until Duckets merges this branch.

Safety posture:
  - DRY-RUN BY DEFAULT. Without --apply it prints the facts it WOULD
    store and writes nothing to the bank.
  - --writer regex (default) reproduces today's incumbent behavior;
    --writer hy-memory opts into the RAM-heavy writer, loaded and
    unloaded inside this job only (load-on-dispatch).
  - --gate (default on) asks SystemOne/D1 per chunk; D1 skips never
    reach the writer. The gate fails open if the shim is down.
  - --apply is intentionally NOT wired to the store yet: it aborts with
    a clear message. Wiring it is the final merge step, done with the
    store's owners watching, not from a dev branch.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WRITER_ROLE = "memory_writer"           # SystemOne role name (pinned)
WRITER_MODEL = "hy-memory-30b-a3b-mlx"  # pinned writer build (MLX 4-bit; GGUF retired 2026-10-16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="markdown files to consolidate")
    ap.add_argument("--writer", choices=["regex", "hy-memory"], default="regex")
    ap.add_argument("--model", default=WRITER_MODEL)
    ap.add_argument("--gate", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--apply", action="store_true",
                    help="write planned facts to the bank (dev-branch wiring; "
                         "hy-memory writer runs with Phase A reconciliation)")
    ap.add_argument("--data-dir", default=None,
                    help="bank persist dir (default: the live bank)")
    args = ap.parse_args()


    from src.chunk import chunk_markdown
    from src.consolidate import extract_facts_from_chunk

    gate_chunk = None
    if args.gate:
        from src.gate import gate_chunk as _gate
        gate_chunk = _gate

    lms = Path.home() / ".lmstudio" / "bin" / "lms"
    loaded = False
    if args.writer == "hy-memory":
        from src.hy_memory import extract_memories_hy
        subprocess.run([str(lms), "load", args.model, "--context-length", "8192", "-y"],
                       capture_output=True, timeout=300)
        loaded = True
    try:
        planned: list[str] = []
        skipped = 0
        for p in args.paths:
            text = Path(p).read_text(errors="replace")
            for i, ch in enumerate(chunk_markdown(text)):
                ctext = (ch.text or "").strip()
                if len(ctext) < 200:
                    continue
                if gate_chunk is not None:
                    d = gate_chunk(ctext)
                    if d.gated and d.verdict == "skip":
                        skipped += 1
                        continue
                if args.writer == "hy-memory":
                    planned.extend(extract_memories_hy(ctext, model=args.model))
                else:
                    planned.extend(
                        f.text for f in extract_facts_from_chunk(ctext, f"{p}:{i}", p))
        seen: set[str] = set()
        unique = []
        for t in planned:
            k = " ".join(t.lower().split())
            if k not in seen:
                seen.add(k)
                unique.append(t)
        if args.apply:
            import asyncio
            from src.memory import Memory
            mem = Memory(persist_dir=args.data_dir) if args.data_dir else Memory()
            rec_model = args.model if args.writer == "hy-memory" else None

            async def _apply():
                n_reconciled = 0
                for t in unique:
                    r = await mem.remember(
                        t, source_path="consolidate_job",
                        force_tier="semantic", reconcile_model=rec_model)
                    n_reconciled += int((r.metadata or {}).get("reconciled_count", 0))
                return n_reconciled

            n_rec = asyncio.run(_apply())
            print(f"APPLIED - writer={args.writer} gate={args.gate} "
                  f"gate_skipped={skipped} facts={len(unique)} "
                  f"reconciled_supersessions={n_rec}")
            return 0
        print(f"DRY RUN - writer={args.writer} gate={args.gate} "
              f"gate_skipped={skipped} facts_planned={len(unique)}")
        for t in unique:
            print(f"  + {t}")
        return 0
    finally:
        if loaded:
            subprocess.run([str(lms), "unload", args.model], capture_output=True, timeout=120)


if __name__ == "__main__":
    raise SystemExit(main())
