"""
dashboard.py — observability dashboard for the DuckBot brain.

A read-only view of the brain's current state:
  - How many chunks per tier (working/episodic/semantic/procedural)
  - Recent sync activity (last 24h)
  - Graph: entities + relationships
  - Memory blocks: count + total size
  - Quarantine: pending reviews
  - Query latency stats (if logs available)

This is a pure read-only module. No LLM, no API cost, no side effects.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .graph import Graph, DEFAULT_GRAPH_PATH
from .blocks import BlockStore, DEFAULT_BLOCKS_PATH
from .injection_scan import QuarantineStore, DEFAULT_QUARANTINE_PATH


# Optional: store + tier imports (chroma is heavy, so wrap in try/except)
def _try_get_store_stats() -> Optional[dict]:
    """Get tier stats from chroma if available."""
    try:
        from .store import MemoryStore
        store = MemoryStore()
        return store.stats() if hasattr(store, "stats") else None
    except Exception:
        return None


def _get_chroma_stats_directly(persist_dir: Optional[Path] = None) -> Optional[dict]:
    """Directly query chroma for tier counts (no LLM calls)."""
    try:
        from .store import MemoryStore
        store = MemoryStore(persist_dir=persist_dir) if persist_dir else MemoryStore()
        s = store.stats()
        result = {"total": s.total, "by_tier": {}}
        for attr in ("working", "episodic", "semantic", "procedural"):
            result["by_tier"][attr] = getattr(s, attr, 0)
        return result
    except Exception as e:
        return {"error": str(e)}


@dataclass
class DashboardReport:
    """The full dashboard report."""
    generated_at: float = field(default_factory=time.time)
    chroma: Optional[dict] = None
    graph: Optional[dict] = None
    blocks: Optional[dict] = None
    quarantine: Optional[dict] = None
    recent_sync: list[dict] = field(default_factory=list)
    last_24h_stats: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "generated_at_iso": datetime.fromtimestamp(
                self.generated_at, tz=timezone.utc
            ).isoformat(),
            "chroma": self.chroma,
            "graph": self.graph,
            "blocks": self.blocks,
            "quarantine": self.quarantine,
            "recent_sync": self.recent_sync,
            "last_24h_stats": self.last_24h_stats,
        }


def build_report(
    graph_path: Optional[Path] = None,
    blocks_path: Optional[Path] = None,
    quarantine_path: Optional[Path] = None,
    chroma_path: Optional[Path] = None,
    watcher_log: Optional[Path] = None,
    last_n_sync: int = 10,
    now: Optional[float] = None,
) -> DashboardReport:
    """Build a dashboard report from the live state."""
    r = DashboardReport()

    # ---- Chroma / tier counts ----
    try:
        persist_dir = Path(chroma_path) if chroma_path else None
        r.chroma = _get_chroma_stats_directly(persist_dir=persist_dir)
    except Exception as e:
        r.chroma = {"error": str(e)}

    # ---- Knowledge graph ----
    try:
        gp = Path(graph_path) if graph_path else DEFAULT_GRAPH_PATH
        if gp.exists():
            with Graph(path=gp) as g:
                r.graph = g.stats()
        else:
            r.graph = {"entities": 0, "relationships": 0, "note": "graph.db not found"}
    except Exception as e:
        r.graph = {"error": str(e)}

    # ---- Memory blocks ----
    try:
        bp = Path(blocks_path) if blocks_path else DEFAULT_BLOCKS_PATH
        if bp.exists():
            with BlockStore(path=bp) as s:
                r.blocks = s.stats()
        else:
            r.blocks = {"blocks": 0, "note": "blocks.db not found"}
    except Exception as e:
        r.blocks = {"error": str(e)}

    # ---- Quarantine ----
    try:
        qp = Path(quarantine_path) if quarantine_path else DEFAULT_QUARANTINE_PATH
        if qp.exists():
            with QuarantineStore(path=qp) as q:
                r.quarantine = q.stats()
        else:
            r.quarantine = {"total": 0, "note": "quarantine.db not found"}
    except Exception as e:
        r.quarantine = {"error": str(e)}

    # ---- Recent sync activity ----
    if watcher_log and Path(watcher_log).exists():
        try:
            # Read tail-only (last ~200 lines) so a multi-GB watcher.log
            # doesn't load into memory. The previous read_text() + split()
            # loaded the entire file before slicing — OOM risk on long runs.
            all_lines = _tail_lines(Path(watcher_log), max_lines=200 + last_n_sync)
            r.recent_sync = _parse_watcher_log(all_lines[-200:])[-last_n_sync:]
            r.last_24h_stats = _summarize_last_24h(
                _parse_watcher_log(all_lines), now=now
            )
        except Exception as e:
            r.recent_sync = [{"error": str(e)}]
    else:
        r.recent_sync = []

    return r


def _tail_lines(path: Path, max_lines: int) -> list[str]:
    """Return the last `max_lines` lines of `path` without loading it whole.

    Reads in 64 KiB chunks from the end and splits on newlines, dropping a
    partial trailing line. Good enough for watcher.log at any size.
    """
    CHUNK = 64 * 1024
    try:
        size = path.stat().st_size
    except OSError:
        return []
    if size == 0:
        return []
    lines: list[bytes] = []
    buf = b""
    with path.open("rb") as f:
        pos = size
        while pos > 0 and len(lines) <= max_lines:
            read = min(CHUNK, pos)
            pos -= read
            f.seek(pos)
            buf = f.read(read) + buf
            parts = buf.split(b"\n")
            buf = parts[0]
            lines = parts[1:] + lines
    # If we read the start of the file, `buf` holds the head (the part before
    # the first newline) — prepend it. If we didn't reach pos 0, `buf` is a
    # partial first line that we must drop.
    if pos == 0 and buf:
        lines = [buf] + lines
    # Drop a trailing empty line from a file ending in '\n'.
    if lines and lines[-1] == b"":
        lines = lines[:-1]
    return [ln.decode("utf-8", errors="replace") for ln in lines[-max_lines:]]


def _parse_watcher_log(lines: list[str]) -> list[dict]:
    """Parse watcher.log lines into structured events."""
    out = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Lines look like: [2026-06-23T13:17:28-0400]   added 11 chunks from /path
        m_ts = line.lstrip("[").split("]")[0] if line.startswith("[") else None
        rest = line.split("] ", 1)[1] if "] " in line else line
        entry: dict = {"raw": line}
        if m_ts:
            entry["ts"] = m_ts
        if "sync pass:" in rest:
            # Sync summary line
            try:
                summary = rest.split("sync pass:", 1)[1].strip()
                # Parse the dict-like representation
                d = json.loads(summary.replace("'", '"'))
                entry["type"] = "sync"
                entry["summary"] = d
            except Exception:
                entry["type"] = "sync"
                entry["summary_raw"] = rest
        elif "chunks from" in rest:
            try:
                # e.g. "added 11 chunks from /path"
                action_part, path_part = rest.split(" chunks from ", 1)
                action_words = action_part.split()
                entry["type"] = "file"
                entry["action"] = action_words[0]
                entry["chunks"] = int(action_words[1]) if len(action_words) > 1 else 0
                entry["path"] = path_part
            except Exception:
                entry["type"] = "file"
                entry["raw_rest"] = rest
        else:
            entry["type"] = "other"
        out.append(entry)
    return out


def _summarize_last_24h(events: list[dict], now: Optional[float] = None) -> dict:
    """Summarize watcher activity in the last 24 hours."""
    now = time.time() if now is None else now
    cutoff = now - 86400
    syncs = 0
    files_processed: set[str] = set()
    chunks_added = 0
    chunks_updated = 0
    errors = 0
    for e in events:
        ts_str = e.get("ts", "")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str).timestamp()
        except Exception:
            continue
        if ts < cutoff:
            continue
        if e.get("type") == "sync":
            syncs += 1
            summary = e.get("summary", {})
            chunks_added += summary.get("added", 0) or 0
            chunks_updated += summary.get("updated", 0) or 0
            errors += len(summary.get("errors", []) or [])
        elif e.get("type") == "file":
            files_processed.add(e.get("path", ""))
    return {
        "syncs": syncs,
        "unique_files_processed": len(files_processed),
        "chunks_added": chunks_added,
        "chunks_updated": chunks_updated,
        "errors": errors,
    }


def format_report(r: DashboardReport) -> str:
    """Format the report as a human-readable string."""
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("🧠 DuckBot Brain Dashboard")
    iso = datetime.fromtimestamp(r.generated_at, tz=timezone.utc).isoformat()
    lines.append(f"   Generated: {iso}")
    lines.append("=" * 60)
    lines.append("")

    # Chroma / tiers
    lines.append("📚 Vector Store (Chroma)")
    if r.chroma is None or "error" in r.chroma:
        lines.append(f"   ⚠️  {r.chroma.get('error', 'unavailable') if r.chroma else 'unavailable'}")
    else:
        lines.append(f"   Total chunks: {r.chroma.get('total', '?')}")
        if "by_tier" in r.chroma:
            for tier, count in r.chroma["by_tier"].items():
                lines.append(f"     {tier}: {count}")
    lines.append("")

    # Graph
    lines.append("🕸️  Knowledge Graph (temporal)")
    if r.graph is None or "error" in r.graph:
        lines.append(f"   ⚠️  {r.graph.get('error', 'unavailable') if r.graph else 'unavailable'}")
    else:
        lines.append(f"   Entities: {r.graph.get('entities', 0)}")
        lines.append(f"   Relationships: {r.graph.get('relationships', 0)}")
        lines.append(f"     active:  {r.graph.get('active_relationships', 0)}")
        lines.append(f"     ended:   {r.graph.get('ended_relationships', 0)}")
        if "entities_by_kind" in r.graph and r.graph["entities_by_kind"]:
            lines.append("   Entities by kind:")
            for kind, n in sorted(r.graph["entities_by_kind"].items()):
                lines.append(f"     {kind}: {n}")
    lines.append("")

    # Blocks
    lines.append("📦 Memory Blocks (Letta-style)")
    if r.blocks is None or "error" in r.blocks:
        lines.append(f"   ⚠️  {r.blocks.get('error', 'unavailable') if r.blocks else 'unavailable'}")
    else:
        lines.append(f"   Blocks: {r.blocks.get('blocks', 0)}")
        lines.append(f"   Total writes (all-time): {r.blocks.get('total_writes', 0)}")
        lines.append(f"   Total chars: {r.blocks.get('total_chars', 0)}")
    lines.append("")

    # Quarantine
    lines.append("🛡️  Injection Quarantine (OWASP ASI06)")
    if r.quarantine is None or "error" in r.quarantine:
        lines.append(f"   ⚠️  {r.quarantine.get('error', 'unavailable') if r.quarantine else 'unavailable'}")
    else:
        lines.append(f"   Total: {r.quarantine.get('total', 0)}")
        lines.append(f"     pending:  {r.quarantine.get('pending', 0)}")
        lines.append(f"     approved: {r.quarantine.get('approved', 0)}")
        lines.append(f"     rejected: {r.quarantine.get('rejected', 0)}")
        by_sev = r.quarantine.get("by_severity", {})
        if by_sev:
            lines.append("   By severity:")
            for sev, n in sorted(by_sev.items()):
                lines.append(f"     severity {sev}: {n}")
    lines.append("")

    # 24h stats
    lines.append("📊 Last 24h activity")
    s = r.last_24h_stats
    if not s:
        lines.append("   (no watcher log parsed)")
    else:
        lines.append(f"   Sync passes: {s.get('syncs', 0)}")
        lines.append(f"   Unique files processed: {s.get('unique_files_processed', 0)}")
        lines.append(f"   Chunks added: {s.get('chunks_added', 0)}")
        lines.append(f"   Chunks updated: {s.get('chunks_updated', 0)}")
        if s.get("errors", 0):
            lines.append(f"   ⚠️  Errors: {s['errors']}")
        else:
            lines.append("   Errors: 0")
    lines.append("")

    # Recent sync events
    if r.recent_sync:
        lines.append(f"📋 Recent sync events (last {len(r.recent_sync)})")
        for e in r.recent_sync[-5:]:
            ts = e.get("ts", "?")[:19]
            if e.get("type") == "sync":
                summary = e.get("summary", {})
                lines.append(
                    f"   {ts} sync: +{summary.get('added', 0)} "
                    f"~{summary.get('updated', 0)} "
                    f"errors={len(summary.get('errors', []) or [])}"
                )
            elif e.get("type") == "file":
                lines.append(
                    f"   {ts} {e.get('action', '?')} {e.get('chunks', 0)} chunks: "
                    f"{e.get('path', '?')[:60]}"
                )
            else:
                lines.append(f"   {ts} {e.get('raw', '?')[:80]}")
    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
