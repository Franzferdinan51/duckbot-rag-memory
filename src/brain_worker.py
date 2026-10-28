"""DuckBot brain worker — the Brain, served over stdio line-JSON.

Why this exists (v0.16.0): the Hermes gateway runs an isolated bundled
Python (``-I``: no user site-packages, PYTHONPATH stripped) that cannot
import chromadb, so the duckbot-brain MemoryProvider could never run
in-process there and had to self-disable (the 2026-10-19 host gate).
The provider now spawns THIS module under the repo venv Python — the
same interpreter the MCP server uses — and forwards its Brain calls
here, one JSON object per line in each direction:

    -> {"id": 1, "op": "recall", "args": {"query": "...", "k": 3}}
    <- {"id": 1, "ok": true, "result": [...]}

Ops: ping, dispatch (src.extensions.tools.dispatch), recall, remember,
wake_up, stats, shutdown. All diagnostics go to stderr; stdout carries
ONLY protocol lines.

Run it exactly like the MCP server:
    .venv/bin/python -m src.brain_worker

Test hook: DUCKBOT_BRAIN_DATA_DIR points the Brain at an alternate data
directory (graph/blocks/quarantine/chroma under it) so tests never
touch the live store. Unset in production — the Brain defaults apply.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Optional

# Repo root on sys.path (this file lives in <root>/src/).
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.connectors.base import Brain  # noqa: E402
from src.extensions import tools as _tools  # noqa: E402

_BRAIN: Optional[Brain] = None


def _get_brain() -> Brain:
    global _BRAIN
    if _BRAIN is None:
        data_dir = os.environ.get("DUCKBOT_BRAIN_DATA_DIR", "")
        if data_dir:
            dd = Path(data_dir)
            dd.mkdir(parents=True, exist_ok=True)
            _BRAIN = Brain(
                graph_path=dd / "graph.db",
                blocks_path=dd / "blocks.db",
                quarantine_path=dd / "quarantine.db",
                persist_dir=dd / "chroma",
            )
        else:
            _BRAIN = Brain()
    return _BRAIN


def _serialize_recall(results) -> list:
    """RecallResult -> JSON dict (same shape as extensions.tools)."""
    out = []
    for r in results:
        out.append({
            "chunk_id": getattr(r, "chunk_id", ""),
            "text": getattr(r, "text", ""),
            "tier": getattr(r, "tier", None),
            "importance": float(getattr(r, "importance", 0.0) or 0.0),
            "score": float(getattr(r, "score", 0.0) or 0.0),
            "source_path": getattr(r, "source_path", "") or "",
            "metadata": getattr(r, "metadata", {}) or {},
        })
    return out


def _handle(op: str, args: dict) -> Any:
    if op == "ping":
        import chromadb
        return {"pong": True, "chromadb": chromadb.__version__}
    if op == "dispatch":
        return _tools.dispatch(args["name"], args.get("args") or {})
    brain = _get_brain()
    if op == "recall":
        kwargs = {"k": int(args.get("k", 5))}
        for key in ("tier", "min_importance", "rerank", "decay",
                    "tier_priors", "fsrs", "skip_superseded", "timeout_ms"):
            if args.get(key) is not None:
                kwargs[key] = args[key]
        return _serialize_recall(brain.recall(args["query"], **kwargs))
    if op == "remember":
        result = brain.remember(
            text=args["text"],
            source_path=args.get("source_path", "<connector>"),
            force_tier=args.get("force_tier"),
        )
        return result.to_dict()
    if op == "wake_up":
        return brain.wake_up(**(args or {}))
    if op == "stats":
        return _tools._serialize_stats(brain.stats())
    raise ValueError(f"unknown op: {op}")


def main() -> int:
    print("[brain-worker] ready", file=sys.stderr, flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        req_id = req.get("id")
        op = req.get("op", "")
        try:
            result = _handle(op, req.get("args") or {})
            resp = {"id": req_id, "ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc(file=sys.stderr)
            resp = {"id": req_id, "ok": False,
                    "error": f"{type(exc).__name__}: {exc}"}
        sys.stdout.write(json.dumps(resp, default=str) + "\n")
        sys.stdout.flush()
        if op == "shutdown":
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
