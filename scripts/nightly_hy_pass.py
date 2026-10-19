#!/usr/bin/env python3
"""
nightly_hy_pass.py — Hy-Memory consolidation pass for the live brain.

Sleep-time consolidation with the Hy-Memory writer (load-on-dispatch):
  1. Load the pinned Hy-Memory build into LM Studio (8k context).
  2. reflect(): recent episodic chunks -> Hy-Memory extraction -> dedupe
     -> promote to the semantic tier. Fails open per chunk to regex
     extraction (extract_memories_hy returns [] on any writer failure).
  3. Unload the writer. The chat model + embedder stay loaded, so a
     running agent session is never disturbed (128GB unified memory
     hosts both comfortably).

Usage (from the repo root, venv):
    .venv/bin/python scripts/nightly_hy_pass.py [--days 30] [--max-chunks 200]
    .venv/bin/python scripts/nightly_hy_pass.py --no-load   # writer already loaded

The promotion path is the normal remember() pipeline with
force_tier=semantic, so Phase A Hy-Memory reconciliation applies when a
reconcile model is configured. Nothing here touches the chat model.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_MODEL = "hy-memory-30b-a3b-mlx"  # MLX 4-bit (GGUF retired 2026-10-16)
LMS = Path.home() / ".lmstudio" / "bin" / "lms"


def _lms(*args: str) -> tuple[int, str]:
    """Run an lms subcommand; return (exit_code, combined_output)."""
    try:
        done = subprocess.run(
            [str(LMS), *args],
            capture_output=True, text=True, timeout=600,
        )
        return done.returncode, (done.stdout + done.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, f"lms {args} failed: {exc}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=30,
                    help="episodic lookback window (default 30)")
    ap.add_argument("--max-chunks", type=int, default=200)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-load", action="store_true",
                    help="writer is already loaded; skip load/unload")
    ap.add_argument("--json", action="store_true",
                    help="full JSON result (default: one summary line)")
    args = ap.parse_args()

    loaded_by_us = False
    if not args.no_load:
        code, out = _lms("load", args.model, "--context-length", "8192", "-y")
        if code != 0:
            print(json.dumps({
                "error": "writer load failed",
                "lms": out[:400],
                "fallback": "regex extraction only (no Hy-Memory)",
            }, indent=2))
            return 1
        loaded_by_us = True
        print(f"[hy-pass] loaded {args.model}")

    from src.memory import Memory
    from src.hy_memory import extract_memories_hy

    def hy_extract(chunk_text: str, chunk_id: str, source_path: str) -> list[str]:
        # Fail-open: extract_memories_hy returns [] on writer failure;
        # reflect() then falls back to regex extraction for that chunk.
        return extract_memories_hy(chunk_text, model=args.model, timeout=180.0)

    import asyncio
    t0 = time.time()
    result = asyncio.run(
        Memory().reflect(
            lookback_days=args.days,
            max_chunks=args.max_chunks,
            extract_callback=hy_extract,
        )
    )

    if loaded_by_us:
        code, out = _lms("unload", args.model)
        if code != 0:
            print(f"[hy-pass] WARN: unload failed: {out[:200]}")
        else:
            print(f"[hy-pass] unloaded {args.model}")

    result["writer"] = args.model
    result["writer_loaded_by_us"] = loaded_by_us
    result["elapsed_s"] = round(time.time() - t0, 1)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        # One-line form for cron delivery (stdout is posted verbatim).
        print(
            f"hy-pass: scanned={result.get('scanned', 0)} "
            f"extracted={result.get('extracted', 0)} "
            f"promoted={result.get('promoted', 0)} "
            f"elapsed={result['elapsed_s']}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
