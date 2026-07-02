"""
dialect.py — AAAK compression dialect for the brain's index layer.

Inspired by MemPalace's "AAAK compression" (compact symbolic format
that lets an LLM scan thousands of entries and know which drawer to
open). The point: an LLM has 200K-token context, but reading 5000
full memory chunks burns most of that. A compact one-line-per-entry
summary lets the LLM reason about the corpus in <500 tokens and then
ask the brain to expand only the few entries it actually needs.

Format per entry (one line):
    @tier:importance score "preview" src=/path/to/source.md

Example (3 chunks):
    working:0.7 "GPU driver issue" src=/notes/2026-06-22.md
    semantic:0.5 "Kai joined project Orion" src=/MEMORY.md
    episodic:0.3 "Friday standup notes" src=/notes/2026-06-20.md

Aggregate format (whole-corpus header):
    # brain index v1 | tiers: w=5 e=12 s=8 p=2 | total=27 chunks

Cost: ~30-60 tokens for 5000 entries (well under the 170-token
budget MemPalace reports for their AAAK variant). Pure ASCII so it
survives any transport.

The dialect is read by humans too — print it to stdout during
debugging and you get an instant visual map of the brain.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence


# One entry, one line.
ENTRY_TEMPLATE = '{tier_short}:{imp:.2f} "{preview}" src={src}'
HEADER_TEMPLATE = "# brain index v1 | tiers: {tier_counts} | total={total} chunks"


def _short_tier(tier: str) -> str:
    """Tier -> single-letter (w/e/s/p). Unknown tiers keep their first char."""
    t = (tier or "").strip().lower()
    return {"working": "w", "episodic": "e", "semantic": "s", "procedural": "p"}.get(
        t, t[:1] or "?"
    )


def compress_chunk(
    text: str,
    tier: str = "unknown",
    importance: float = 0.0,
    source_path: str = "",
    preview_chars: int = 80,
) -> str:
    """Compress a single chunk to one dialect line.

    Args:
        text: the chunk's text (we take a preview, not the whole thing).
        tier: working / episodic / semantic / procedural.
        importance: 0..1.
        source_path: where this came from; the agent can use this to
            ask the brain to expand the full text later.
        preview_chars: how many chars of text to include.
    """
    text = (text or "").strip().replace("\n", " ").replace("\r", " ")
    # Collapse repeated whitespace to keep the line readable.
    import re
    text = re.sub(r"\s+", " ", text)
    preview = text[:preview_chars]
    if len(text) > preview_chars:
        preview += "…"
    # Escape double quotes inside the preview.
    preview = preview.replace('"', '\\"')
    src = source_path or "<unknown>"
    # Importance should be 0..1, but stored values can drift (the recall
    # loop adds 0.02 per recall without an upper cap in some code paths).
    # Clamp here so the dialect never shows a misleading 137.60.
    imp = max(0.0, min(1.0, float(importance or 0.0)))
    return ENTRY_TEMPLATE.format(
        tier_short=_short_tier(tier),
        imp=imp,
        preview=preview,
        src=src,
    )


def compress_corpus(
    chunks: Iterable[dict],
    max_chunks: int = 5000,
    preview_chars: int = 80,
) -> str:
    """Compress a corpus of chunks to a single string the LLM can scan.

    Each input chunk is a dict with keys: text, tier, importance, source_path.
    Output is a header + one line per chunk. Stops at `max_chunks` to
    keep the dialect bounded — the brain is the authority on the rest.
    """
    lines: list[str] = []
    counts: dict[str, int] = {}
    for i, c in enumerate(chunks):
        if i >= max_chunks:
            break
        tier = c.get("tier", "unknown")
        counts[tier] = counts.get(tier, 0) + 1
        lines.append(compress_chunk(
            text=c.get("text", ""),
            tier=tier,
            importance=c.get("importance", 0.0),
            source_path=c.get("source_path", ""),
            preview_chars=preview_chars,
        ))
    tier_counts = " ".join(f"{_short_tier(k)}={v}" for k, v in sorted(counts.items()))
    header = HEADER_TEMPLATE.format(tier_counts=tier_counts or "none", total=len(lines))
    return header + "\n" + "\n".join(lines)


def parse_entry(line: str) -> Optional[dict]:
    """Parse a single dialect line back into a dict. Used by tests and
    by the brain itself when it needs to recover the chunk_id from a
    referenced entry (the LLM picks a line, the brain opens the drawer).
    """
    if not line or line.startswith("#"):
        return None
    try:
        # Format: w:0.70 "preview text" src=/path
        # Split on first ':' then on first space.
        tier_part, rest = line.split(":", 1)
        imp_part, rest = rest.split(" ", 1)
        importance = float(imp_part)
        # Find the quoted preview and the src= part.
        first_q = rest.find('"')
        if first_q < 0:
            return None
        # Find the matching close quote (we escaped inner quotes so the
        # last quote is the closer).
        last_q = rest.rfind('"')
        if last_q <= first_q:
            return None
        preview = rest[first_q + 1:last_q].replace('\\"', '"')
        src_part = rest[last_q + 1:].strip()
        if src_part.startswith("src="):
            src_part = src_part[4:]
        return {
            "tier": tier_part,
            "importance": importance,
            "preview": preview,
            "source_path": src_part,
        }
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# INDEX.md generation — article technique: "one INDEX.md per vault concern"
#
# From @rohit4verse / @wandermist "I Made My Hermes Agent 10x Faster":
#   "Structure your vault with one INDEX.md per concern. Each INDEX.md
#    acts as a high-signal table of contents for that topic area, giving
#    the embedding model a coherent anchor to retrieve against."
#
# DuckBot applies this as: generate a structured index of the entire corpus
# grouped by CoALA tier + source file. The index itself becomes a synthetic
# high-value chunk that anchors retrieval around "what memory tiers exist",
# "what files are in the system", and "what topics are covered" queries.
# ---------------------------------------------------------------------------

def emit_index_md(source_dir: str = "data/brain_export.md") -> str:
    """Generate a structured INDEX.md for the corpus.

    Groups brain_export.md chunks by tier and source file, then emits
    a navigable markdown index. This is the "one INDEX.md per vault concern"
    pattern from the Hermes speed-up article, applied at the corpus level.

    Returns the full index text. Caller saves it or prints it.
    """
    from pathlib import Path
    import re

    export_path = Path(source_dir)
    if not export_path.exists():
        return f"# INDEX.md\n\n> Source not found: {source_dir}\n> Run `duck-memory export` first.\n"

    text = export_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Parse ## Tier sections
    current_tier = None
    current_section_lines: list[str] = []
    sections: dict[str, list[str]] = {}

    for line in lines:
        tier_match = re.match(r"^## (\w+)$", line)
        if tier_match:
            if current_tier:
                sections[current_tier] = current_section_lines
            current_tier = tier_match.group(1)
            current_section_lines = []
        else:
            current_section_lines.append(line)

    if current_tier:
        sections[current_tier] = current_section_lines

    # Parse chunks within each tier section
    # Format: ### <chunk_id>  (tier=<tier>, importance=<n>)
    chunks_by_tier: dict[str, list[dict]] = {}

    for tier, section_lines in sections.items():
        chunks: list[dict] = []
        i = 0
        while i < len(section_lines):
            line = section_lines[i]
            chunk_match = re.match(r"^### (.+?)  \(tier=(\w+), importance=([\d.]+)\)$", line)
            if chunk_match:
                chunk_id = chunk_match.group(1)
                chunk_tier = chunk_match.group(2)
                importance = float(chunk_match.group(3))
                # Collect content lines until next ### or end
                content_lines = []
                j = i + 1
                while j < len(section_lines):
                    next_line = section_lines[j]
                    if next_line.startswith("### "):
                        break
                    if next_line.startswith("_source:") or next_line.startswith("  "):
                        content_lines.append(next_line.strip())
                    j += 1
                content = " ".join(content_lines).strip()
                # Extract source file
                src_match = re.search(r"_source: (.+)$", "\n".join(section_lines[i+1:i+3]))
                source = src_match.group(1) if src_match else "unknown"
                chunks.append({
                    "id": chunk_id,
                    "importance": importance,
                    "source": source,
                    "preview": content[:120] + "..." if len(content) > 120 else content,
                })
            i += 1
        chunks_by_tier[tier] = chunks

    # Build INDEX.md
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines_out = [
        "# INDEX.md — DuckBot Memory Corpus Index",
        "",
        f"_Generated {now}_",
        f"_Source: {source_dir}_",
        "",
        "## Purpose",
        "",
        "This index summarizes the entire memory corpus. Use it as a",
        "high-signal anchor for queries like \"what topics exist in memory?\"",
        "or \"what rules has the system learned?\".",
        "",
        "## Corpus Summary",
        "",
    ]

    total = sum(len(v) for v in chunks_by_tier.values())
    lines_out.append(f"- **Total chunks:** {total}")
    for tier, chunks in chunks_by_tier.items():
        lines_out.append(f"  - {tier.capitalize()}: {len(chunks)} chunks")

    # Sources
    all_sources = set()
    for tier_chunks in chunks_by_tier.values():
        for c in tier_chunks:
            src = c["source"]
            # Normalize Windows paths for display
            display = src.replace("\\", "/").split("/")[-1]
            all_sources.add(display)

    lines_out.extend([
        "",
        "## Source Files",
        "",
    ])
    for src in sorted(all_sources):
        lines_out.append(f"- `/{src}`")

    # Tier-by-tier index
    tier_labels = {
        "semantic": "🔹 Semantic — Distilled facts, entities, user preferences",
        "procedural": "🔧 Procedural — Rules, behavioral norms, patterns",
        "episodic": "📅 Episodic — Session logs, dated events, decisions",
        "working": "⚡ Working — Today's active session",
    }

    for tier in ["semantic", "procedural", "episodic", "working"]:
        chunks = chunks_by_tier.get(tier, [])
        if not chunks:
            continue
        label = tier_labels.get(tier, tier.capitalize())
        lines_out.extend(["", f"## {label}", ""])
        # Group by source
        by_source: dict[str, list[dict]] = {}
        for c in chunks:
            src = c["source"].replace("\\", "/").split("/")[-1]
            by_source.setdefault(src, []).append(c)
        for src, src_chunks in sorted(by_source.items()):
            lines_out.append(f"### /{src}")
            for c in sorted(src_chunks, key=lambda x: -x["importance"])[:5]:
                importance_bar = "▓" * int(c["importance"] * 10) + "░" * (10 - int(c["importance"] * 10))
                lines_out.append(f"- [{importance_bar}] {c['preview']}")

    lines_out.extend(["", "---", "_End of INDEX.md_"])
    return "\n".join(lines_out)
