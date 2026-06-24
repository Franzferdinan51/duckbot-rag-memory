"""
connectors/dreaming.py — bidirectional bridge between the brain and
OpenClaw's existing dreaming subsystem.

OpenClaw dreaming surface:
    - Diary:   ~/.openclaw/workspace/DREAMS.md
    - Storage: ~/.openclaw/workspace/memory/dreaming/{deep,light,rem}/<date>.md

The bridge does two things:

  read()  — Pull new entries from DREAMS.md + memory/dreaming/*.md and
            ingest them into the brain as `semantic` tier (consolidated
            wisdom). Skips entries we've already ingested (idempotent via
            a state file).

  cycle() — Run a consolidation pass: pick high-importance episodic
            chunks, distill them to a small set of `semantic` rules, and
            write a new entry to memory/dreaming/deep/<date>.md so
            OpenClaw's dreamer picks it up on its next pass.

This is NOT a re-implementation of OpenClaw's dreaming; it consumes and
produces to the same files OpenClaw's dreamer already uses.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..memory import Memory, Tier


# Default locations — match the OpenClaw install you showed.
DEFAULT_DREAMS_DIARY = Path.home() / ".openclaw" / "workspace" / "DREAMS.md"
DEFAULT_DREAMING_DIR = Path.home() / ".openclaw" / "workspace" / "memory" / "dreaming"
DEFAULT_STATE_PATH = Path.home() / ".openclaw" / "workspace" / "memory" / "dreaming" / ".brain_state.json"


@dataclass
class DreamIngestResult:
    """Result of a read() pass."""
    new_entries: int = 0
    skipped: int = 0
    by_kind: dict = field(default_factory=dict)  # {"deep": 2, "light": 1, "rem": 0, "diary": 1}
    sources: list = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "new_entries": self.new_entries,
            "skipped": self.skipped,
            "by_kind": self.by_kind,
            "sources": self.sources,
            "error": self.error,
        }


@dataclass
class DreamCycleResult:
    """Result of a cycle() pass (write-side)."""
    distilled_chunks: int = 0
    by_tier: dict = field(default_factory=dict)
    output_files: list = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "distilled_chunks": self.distilled_chunks,
            "by_tier": self.by_tier,
            "output_files": self.output_files,
            "error": self.error,
        }


class DreamingBridge:
    """Bidirectional bridge between the brain and OpenClaw's dreaming surface.

    Usage:
        bridge = DreamingBridge(memory=Memory(...))
        ingest = bridge.read()      # pull DREAMS.md + memory/dreaming/*.md -> brain
        cycle  = bridge.cycle()     # pull high-importance episodic -> dream entry
    """

    def __init__(
        self,
        memory: Memory,
        dreams_diary: Path = DEFAULT_DREAMS_DIARY,
        dreaming_dir: Path = DEFAULT_DREAMING_DIR,
        state_path: Path = DEFAULT_STATE_PATH,
    ):
        self.memory = memory
        self.dreams_diary = Path(dreams_diary)
        self.dreaming_dir = Path(dreaming_dir)
        self.state_path = Path(state_path)
        self._state = self._load_state()

    # ------------------------------------------------------------------
    # State — track which entries we've already ingested (idempotent).
    # ------------------------------------------------------------------

    def _load_state(self) -> dict:
        if not self.state_path.exists():
            return {"ingested": {}, "last_cycle": 0.0}
        try:
            with self.state_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"ingested": {}, "last_cycle": 0.0}

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2, sort_keys=True)
        tmp.replace(self.state_path)

    def _key_for(self, source: str, content: str) -> str:
        """Stable hash so we can tell if an entry changed since last ingest."""
        h = hashlib.sha256()
        h.update(source.encode("utf-8"))
        h.update(b"\0")
        h.update(content.encode("utf-8"))
        return h.hexdigest()[:16]

    # ------------------------------------------------------------------
    # READ — pull from DREAMS.md + memory/dreaming/*.md into the brain.
    # ------------------------------------------------------------------

    async def read(self) -> DreamIngestResult:
        """Walk the dreaming surface and ingest new entries into the brain.

        Idempotent: uses content-hash state to skip already-ingested entries.
        """
        result = DreamIngestResult()
        ingested = self._state.setdefault("ingested", {})

        sources = []
        if self.dreams_diary.exists():
            sources.append(("diary", self.dreams_diary))
        if self.dreaming_dir.exists():
            for sub in ("deep", "light", "rem"):
                sub_dir = self.dreaming_dir / sub
                if sub_dir.exists():
                    for f in sorted(sub_dir.glob("*.md")):
                        sources.append((sub, f))

        for kind, path in sources:
            try:
                content = path.read_text(encoding="utf-8")
            except OSError as e:
                result.error = f"read error {path}: {e}"
                continue

            # Skip OpenClaw's metadata block (it's not dream content).
            # DREAMS.md format:
            #   <!-- openclaw:dreaming:diary:start -->
            #   ---
            #   *<date>*\n\n<dream text>\n\n---\n
            entries = self._split_entries(content)
            for entry in entries:
                key = self._key_for(str(path), entry)
                if key in ingested:
                    result.skipped += 1
                    continue
                await self.memory.remember(
                    entry,
                    source_path=f"<dreaming/{kind}>{path.name}",
                    force_tier="semantic",  # dreams = consolidated wisdom
                    metadata={"dream_kind": kind, "source_file": str(path)},
                )
                ingested[key] = time.time()
                result.new_entries += 1
                result.by_kind[kind] = result.by_kind.get(kind, 0) + 1
                result.sources.append(f"{kind}:{path.name}")

        self._save_state()
        return result

    @staticmethod
    def _split_entries(content: str) -> list[str]:
        """Split a dream diary into individual dream entries.

        DREAMS.md entries are separated by `\\n---\\n` and each starts with
        `*<date>*`. We strip the metadata header and split on the separator.
        """
        # Strip the marker comment block.
        content = re.sub(
            r"<!--\s*openclaw:dreaming:diary:start\s*-->",
            "",
            content,
        )
        # Split on horizontal rule.
        chunks = re.split(r"\n---\s*\n", content)
        out = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk or len(chunk) < 40:
                continue  # skip blank / fragment chunks
            out.append(chunk)
        return out

    # ------------------------------------------------------------------
    # WRITE — cycle: episodic -> dream entry.
    # ------------------------------------------------------------------

    async def cycle(self, k: int = 10, min_importance: float = 0.5) -> DreamCycleResult:
        """Pick high-importance episodic chunks and write a dream entry.

        This is the consolidation pass: episodic events get distilled to
        a `semantic` rule and a `dreaming/deep/<date>.md` file is written
        so OpenClaw's dreamer picks it up.

        For now the distillation is deterministic (template-based), not
        LLM-summarized. OpenClaw's dreamer can do LLM synthesis on top.
        """
        result = DreamCycleResult()

        # Sample episodic + procedural chunks via recall() with empty query.
        # This returns the most recent chunks. Importance is read from
        # the result metadata (set at remember() time). Dedupe by chunk_id
        # in case recall() returns overlapping hits across tiers.
        seen: set = set()
        chunks: list = []
        try:
            r = await self.memory.recall(query="", k=k, tier="episodic")
            for hit in r.results:
                cid = getattr(hit, "chunk_id", None) or id(hit)
                if cid not in seen:
                    seen.add(cid)
                    chunks.append(hit)
        except Exception as e:
            result.error = f"recall(episodic) failed: {e}"
            return result
        try:
            r = await self.memory.recall(query="", k=k, tier="procedural")
            for hit in r.results:
                cid = getattr(hit, "chunk_id", None) or id(hit)
                if cid not in seen:
                    seen.add(cid)
                    chunks.append(hit)
        except Exception as e:
            # Non-fatal — just continue with what we have.
            pass

        # Filter by importance. Read from .importance (RecallResult field)
        # with fallback to metadata["importance"].
        keepers = []
        for c in chunks:
            meta = c.metadata if hasattr(c, "metadata") and c.metadata else {}
            imp = getattr(c, "importance", 0.0) or meta.get("importance", 0.0)
            try:
                imp = float(imp)
            except (TypeError, ValueError):
                imp = 0.0
            text = c.text if hasattr(c, "text") else ""
            if imp >= min_importance and text and len(text) > 40:
                keepers.append((c, imp, meta))

        # Group by tier for the output. RecallResult.tier is a string.
        by_tier: dict[str, int] = {}
        for c, _, _ in keepers:
            t = getattr(c, "tier", "unknown") or "unknown"
            by_tier[t] = by_tier.get(t, 0) + 1
        result.distilled_chunks = len(keepers)
        result.by_tier = by_tier

        if not keepers:
            return result

        # Build a dream entry deterministically.
        now = time.time()
        date_str = time.strftime("%Y-%m-%d", time.localtime(now))
        lines = [
            f"# Dream — {date_str}",
            "",
            f"Consolidated {len(keepers)} high-importance chunks from the brain.",
            "",
            "## What stood out",
            "",
        ]
        for c, imp, _ in keepers[:20]:
            t = getattr(c, "tier", "unknown") or "unknown"
            preview = c.text[:280].replace("\n", " ")
            lines.append(f"- **[{t}]** (importance {imp:.2f}) {preview}")
        lines.extend(["", "---", ""])

        out_dir = self.dreaming_dir / "deep"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{date_str}.md"
        # Append-only — don't clobber OpenClaw's own dream entries.
        with out_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines))

        result.output_files.append(str(out_path))
        self._state["last_cycle"] = now
        self._save_state()
        return result


# -----------------------------------------------------------------------------
# Sync wrappers — same `_run_async` pattern as the Brain facade.
# -----------------------------------------------------------------------------

def _run_async(coro):
    import asyncio
    import concurrent.futures
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


def read_dreams(
    memory: Memory,
    dreams_diary: Path = DEFAULT_DREAMS_DIARY,
    dreaming_dir: Path = DEFAULT_DREAMING_DIR,
) -> dict:
    """Sync wrapper: pull dream surface -> brain. Returns dict for MCP/CLI."""
    bridge = DreamingBridge(memory, dreams_diary, dreaming_dir)
    return _run_async(bridge.read()).to_dict()


def write_dream_cycle(
    memory: Memory,
    dreams_diary: Path = DEFAULT_DREAMS_DIARY,
    dreaming_dir: Path = DEFAULT_DREAMING_DIR,
    k: int = 10,
    min_importance: float = 0.5,
) -> dict:
    """Sync wrapper: brain -> dream surface. Returns dict for MCP/CLI."""
    bridge = DreamingBridge(memory, dreams_diary, dreaming_dir)
    return _run_async(bridge.cycle(k=k, min_importance=min_importance)).to_dict()
