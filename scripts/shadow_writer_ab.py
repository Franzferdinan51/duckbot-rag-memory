#!/usr/bin/env python3
"""
shadow_writer_ab.py — Phase 2 shadow A/B: incumbent vs Hy-Memory writer.

Runs both extractors over real brain chunks and writes a comparison
report. THE SHADOW NEVER WRITES TO THE BANK — it imports no store code.

  incumbent : src.consolidate.extract_facts_from_chunk (regex path)
  candidate : src.hy_memory.extract_memories_hy (explicit model only)

Optionally gates candidate chunks through SystemOne/D1 first (--gate):
chunks D1 votes to skip are counted, not written anywhere either way.

Usage:
  python scripts/shadow_writer_ab.py [--files N] [--max-chunks N]
      [--gate] [--model hy-memory-30b-a3b-mlx] [--out PATH] [--no-load]
Loads the writer model with `lms load` and unloads it afterwards
(load-on-dispatch) unless --no-load is given.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunk import chunk_markdown, iter_markdown_files  # noqa: E402
from src.consolidate import extract_facts_from_chunk  # noqa: E402
from src.hy_memory import extract_memories_hy  # noqa: E402

# Both agent homes: whichever agent Duckets is running (Hermes now,
# OpenClaw before) keeps its memory files in its own dir. The shadow
# reads the most recent files across ALL of them (Duckets' order,
# 2026-10-16) so the comparison always runs on the live agent's data.
DEFAULT_MEMORY_DIRS = [
    Path.home() / ".hermes" / "memories",
    Path.home() / ".openclaw" / "workspace" / "memory",
]


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _dup_rate(texts: list[str]) -> float:
    if not texts:
        return 0.0
    seen, dups = set(), 0
    for t in texts:
        n = _norm(t)
        if n in seen:
            dups += 1
        seen.add(n)
    return dups / len(texts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory-dir", action="append", default=None,
                    help="memory dir to read (repeatable); default: all known agent memory dirs")
    ap.add_argument("--corpus", default="",
                    help="single markdown file to use as the corpus instead")
    ap.add_argument("--files", type=int, default=6, help="most recent N memory files")
    ap.add_argument("--max-chunks", type=int, default=36)
    ap.add_argument("--gate", action="store_true", help="run the D1 gate first")
    ap.add_argument("--model", default="hy-memory-30b-a3b-mlx")
    ap.add_argument("--out", default="")
    ap.add_argument("--no-load", action="store_true")
    args = ap.parse_args()

    if args.corpus:
        files = [Path(args.corpus)]
        mem_dirs = []
    else:
        mem_dirs = [Path(d) for d in (args.memory_dir or DEFAULT_MEMORY_DIRS)]
        by_path: dict[Path, Path] = {}
        for d in mem_dirs:
            if d.is_dir():
                for f in d.glob("*.md"):
                    by_path[f.resolve()] = f
        files = sorted(by_path.values(), key=lambda q: q.stat().st_mtime)[-args.files:]
    if not files:
        print(f"no memory files under {[str(d) for d in mem_dirs]}")
        return 1

    chunks: list[tuple[str, str, str]] = []  # (text, chunk_id, source_path)
    for path in files:
        for i, ch in enumerate(chunk_markdown(path.read_text(errors="replace"))):
            text = (ch.text or "").strip()
            if len(text) >= 200:
                chunks.append((text, f"{path.name}:{i}", str(path)))
    chunks = chunks[: args.max_chunks]
    print(f"corpus: {len(files)} files, {len(chunks)} chunks")
    for f in files:
        print(f"  src: {f}")

    gate = None
    if args.gate:
        from src.gate import gate_chunk
        gate = gate_chunk

    lms = Path.home() / ".lmstudio" / "bin" / "lms"
    loaded = False
    if not args.no_load:
        subprocess.run([str(lms), "load", args.model, "--context-length", "8192", "-y"],
                       capture_output=True, timeout=300)
        loaded = True
    try:
        rows = []
        t0 = time.time()
        for text, cid, src_path in chunks:
            inc = [f.text for f in extract_facts_from_chunk(text, cid, src_path)]
            decision = gate(text) if gate else None
            if decision and decision.gated and decision.verdict == "skip":
                hy: list[str] = []
                skipped = True
            else:
                hy = extract_memories_hy(text, model=args.model)
                skipped = False
            rows.append({
                "chunk_id": cid,
                "incumbent": inc,
                "hy_memory": hy,
                "gate": ({"verdict": decision.verdict, "bank": decision.bank,
                          "confidence": decision.confidence, "gated": decision.gated}
                         if decision else None),
                "gate_skipped": skipped,
            })
        elapsed = time.time() - t0
    finally:
        if loaded:
            subprocess.run([str(lms), "unload", args.model], capture_output=True, timeout=120)

    inc_all = [t for r in rows for t in r["incumbent"]]
    hy_all = [t for r in rows for t in r["hy_memory"]]
    inc_norm = {_norm(t) for t in inc_all}
    hy_norm = {_norm(t) for t in hy_all}
    gated_rows = [r for r in rows if r["gate"] and r["gate"]["gated"]]
    report = {
        "sources": [str(f) for f in files],
        "chunks": len(rows),
        "elapsed_s": round(elapsed, 1),
        "incumbent_facts": len(inc_all),
        "hy_facts": len(hy_all),
        "overlap_exact": len(inc_norm & hy_norm),
        "hy_novel": len(hy_norm - inc_norm),
        "incumbent_only": len(inc_norm - hy_norm),
        "hy_dup_rate": round(_dup_rate(hy_all), 3),
        "incumbent_dup_rate": round(_dup_rate(inc_all), 3),
        "hy_empty_chunks": sum(1 for r in rows if not r["hy_memory"] and not r["gate_skipped"]),
        "gate": {
            "enabled": bool(gate),
            "gated_chunks": len(gated_rows),
            "skipped": sum(1 for r in rows if r["gate_skipped"]),
            "banks": {b: sum(1 for r in gated_rows if r["gate"]["bank"] == b)
                      for b in ("chat", "working", "code")},
        } if gate else None,
        "rows": rows,
    }
    out = Path(args.out) if args.out else (
        Path(__file__).resolve().parent.parent / "data" / "shadow" /
        time.strftime("shadow-%Y%m%d-%H%M%S.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=1))
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
