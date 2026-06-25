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
