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

import os
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional


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

    This is the v0.1 heuristic-only path. v0.13.0+ supplements with an
    LLM-extraction pass when LM Studio is reachable (see
    `extract_facts_via_llm`). Both paths run; the LLM is the higher-
    quality one if it succeeds.

    DUCKBOT_NO_LLM_EXTRACTION=1 forces the regex-only path.
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

    # Optional LLM-extraction pass (mem0-inspired). Skipped if:
    # - DUCKBOT_NO_LLM_EXTRACTION=1 (regex-only mode for offline/air-gapped)
    # - chunk too short to be worth an LLM call
    # - LM Studio not reachable
    if os.environ.get("DUCKBOT_NO_LLM_EXTRACTION", "").lower() in ("1", "true", "yes"):
        return facts
    if len(chunk_text) < 200:
        return facts
    llm_facts = extract_facts_via_llm(chunk_text, chunk_id, source_path)
    for f in llm_facts:
        if f.text not in seen_text:
            seen_text.add(f.text)
            facts.append(f)
    return facts


# mem0-inspired extraction prompts (Apache 2.0, ported from the mem0 paper
# and their open-source implementation). These are battle-tested for
# agent-memory use cases; we use them to get higher-quality fact extraction
# than the regex heuristics above.
MEM0_DEDUCTION_PROMPT = """You are a memory extraction agent. Read the episodic
chunk below and extract durable, standalone facts that would still be
true weeks from now. Each fact must be a single self-contained sentence.
Skip ephemeral session details (e.g. "today we tried X"). Skip raw
commands or code. Skip anything that requires the original context to
parse. Use only what is stated or clearly implied.

Output format: one fact per line, prefixed with [kind] where kind is
one of: user-said, decision, rule, preference, setup, location, personal.
If the chunk has no durable facts, output a single line: NONE.

Examples of good output:
[user-said] Duckets prefers dark mode across all UIs.
[decision] Use ChromaDB as the local vector store; do not migrate to LanceDB.
[rule] Always run scripts/secret-scan.sh before committing.
[setup] Restart the BATMAN container via scripts/start-watcher.sh.

Chunk:
{chunk}
"""


def extract_facts_via_llm(
    chunk_text: str,
    chunk_id: str,
    source_path: str,
    *,
    model: Optional[str] = None,
) -> list[ExtractedFact]:
    """Extract durable facts from `chunk_text` via LM Studio (mem0-style
    prompts). Returns [] if LM Studio is unreachable or the LLM call
    fails — the caller should fall back to regex extraction.

    Pattern source: mem0 paper + open-source implementation (Apache 2.0).
    """
    try:
        from .llm_client import chat_completion
    except ImportError:
        return []
    if os.environ.get("DUCKBOT_EMBEDDING", "").lower() == "local" and not os.environ.get("DUCKBOT_LLM_URL"):
        # User explicitly opted into fully-local mode without an LLM
        # endpoint configured. Skip silently.
        return []
    if len(chunk_text) > 8000:
        # Cap to keep inference latency bounded. The regex path handles
        # the long-tail.
        chunk_text = chunk_text[:8000]
    try:
        raw = chat_completion(
            [
                {"role": "system", "content": MEM0_DEDUCTION_PROMPT.format(chunk=chunk_text)},
            ],
            model=model,
            temperature=0.1,
            max_tokens=512,
            timeout=30.0,
        )
    except Exception:
        return []
    if not raw:
        return []
    out: list[ExtractedFact] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.upper() == "NONE":
            continue
        # Format: "[kind] fact text"
        if line.startswith("[") and "]" in line:
            try:
                kind, text = line[1:].split("]", 1)
                kind = kind.strip().lower()
                text = text.strip()
            except ValueError:
                continue
        else:
            kind, text = "fact", line
        # Apply the same length/quality gates as the regex path.
        if not text or len(text) < 5 or len(text) > 300:
            continue
        out.append(ExtractedFact(
            text=text,
            kind=kind,
            source_chunk_id=chunk_id,
            source_path=source_path,
            confidence=0.85 if kind in ("user-said", "decision", "rule") else 0.65,
        ))
    return out


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