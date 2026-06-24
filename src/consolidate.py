"""
consolidate.py — episodic → semantic distillation.

The "dream" pass. Periodically (cron-driven) we:
  1. Pull recent episodic chunks (last 7 days by default)
  2. Group by topic via embeddings clustering (simple: cosine threshold)
  3. For each cluster, ask LLM to extract durable facts
  4. Add extracted facts to the SEMANTIC tier
  5. Optionally mark old episodic chunks as superseded

Pattern from Letta's archival consolidation + mem0's extraction-first approach.
For v0.1 we skip LLM extraction and use simple heuristic fact extraction
(regex patterns for "Duckets said X", "decided X", etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable


# Patterns that signal a "durable fact" worth promoting to semantic memory.
# Inspired by the kind of entries that show up in MEMORY.md as "Added YYYY-MM-DD".
FACT_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?:Duckets|user|he|she)\s+(?:said|told|stated|decided)\s+(?:that\s+)?(.+?)(?:\.|$)", re.IGNORECASE), "user-said"),
    (re.compile(r"(?:we|let's|let us|going to)\s+(?:now\s+)?(.+?)(?:\.|$)", re.IGNORECASE), "decision"),
    (re.compile(r"(?:rule|always|never|must|should not|do not)\s+(.+?)(?:\.|$)", re.IGNORECASE), "rule"),
    (re.compile(r"(?:installed|set up|configured)\s+(.+?)(?:\.|$)", re.IGNORECASE), "setup"),
    (re.compile(r"(?:preference|prefers|likes|favors)\s+(.+?)(?:\.|$)", re.IGNORECASE), "preference"),
    (re.compile(r"(?:address|home|located at|lives at)\s+(.+?)(?:\.|$)", re.IGNORECASE), "location"),
    (re.compile(r"(?:birthday|born on)\s+(.+?)(?:\.|$)", re.IGNORECASE), "personal"),
]


@dataclass
class ExtractedFact:
    text: str
    kind: str       # from FACT_PATTERNS kind
    source_chunk_id: str
    source_path: str
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "kind": self.kind,
            "source_chunk_id": self.source_chunk_id,
            "source_path": self.source_path,
            "confidence": self.confidence,
        }


def extract_facts_from_chunk(
    chunk_text: str,
    chunk_id: str,
    source_path: str,
) -> list[ExtractedFact]:
    """Extract candidate facts from an episodic chunk using regex heuristics.

    This is the v0.1 heuristic-only path. Future: replace with LLM extraction.
    """
    facts: list[ExtractedFact] = []
    seen_text: set[str] = set()

    for pattern, kind in FACT_PATTERNS:
        for m in pattern.finditer(chunk_text):
            text = m.group(1).strip()
            # Strip markdown formatting
            text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
            text = re.sub(r"`([^`]+)`", r"\1", text)
            text = text.strip(" .,;:")
            if not text or len(text) < 5 or len(text) > 300:
                continue
            if text in seen_text:
                continue
            seen_text.add(text)
            facts.append(ExtractedFact(
                text=text,
                kind=kind,
                source_chunk_id=chunk_id,
                source_path=source_path,
                confidence=0.6 if kind in ("user-said", "decision", "rule") else 0.4,
            ))
    return facts


def extract_facts_from_chunks(chunks: Iterable[dict]) -> list[ExtractedFact]:
    """Bulk version. Chunks must have {id, text, metadata.source_path}."""
    all_facts: list[ExtractedFact] = []
    for chunk in chunks:
        facts = extract_facts_from_chunk(
            chunk["text"],
            chunk["id"],
            chunk.get("metadata", {}).get("source_path", "<unknown>"),
        )
        all_facts.extend(facts)
    return all_facts


def deduplicate_facts(facts: list[ExtractedFact], similarity_threshold: float = 0.6) -> list[ExtractedFact]:
    """Naive dedup using word-set Jaccard similarity.

    Real dedup needs embeddings. This is the cheap pre-pass.
    Uses word-level (not char-level) Jaccard so that "use cua-driver" and
    "use cua-driver v0.6.2" share ~67% of words but don't dedup (different facts).
    Threshold 0.6 catches "near-duplicates with minor additions".
    """
    def word_set(s: str) -> set[str]:
        return set(s.lower().split())

    kept: list[ExtractedFact] = []
    for f in facts:
        fset = word_set(f.text)
        is_dup = False
        for k in kept:
            kset = word_set(k.text)
            union = len(fset | kset)
            if union == 0:
                continue
            jaccard = len(fset & kset) / union
            if jaccard >= similarity_threshold:
                is_dup = True
                # Keep the higher-confidence one
                if f.confidence > k.confidence:
                    k.text = f.text
                    k.confidence = f.confidence
                break
        if not is_dup:
            kept.append(f)
    return kept


__all__ = [
    "ExtractedFact",
    "extract_facts_from_chunk",
    "extract_facts_from_chunks",
    "deduplicate_facts",
    "FACT_PATTERNS",
]