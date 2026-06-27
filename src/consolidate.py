"""
consolidate.py — episodic → semantic distillation.

The "dream" pass. Periodically (cron-driven) we:
  1. Pull recent episodic chunks (last 7 days by default)
  2. Group by topic via embeddings clustering (simple: cosine threshold)
3. For each cluster, accept agent-extracted facts when provided, then
   fall back to lightweight regex extraction
  4. Add extracted facts to the SEMANTIC tier
  5. Optionally mark old episodic chunks as superseded

Pattern from Letta's archival consolidation + mem0's extraction-first
approach, but the brain stays lightweight: it never auto-loads a chat
model during reflect(). Agent-supplied facts are preferred; regex
patterns for "Duckets said X", "decided X", etc. are the fallback.
"""

from __future__ import annotations

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


def _extract_via_regex(
    chunk_text: str,
    chunk_id: str,
    source_path: str,
) -> list[ExtractedFact]:
    """Regex-only fact extraction. Used as a fallback when LLM is unavailable
    or as the only path when DUCKBOT_REGEX_ONLY=1 is set."""
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


def extract_facts_from_chunk(
    chunk_text: str,
    chunk_id: str,
    source_path: str,
    *,
    agent_facts: Optional[list[str]] = None,
) -> list[ExtractedFact]:
    """Extract durable facts from a chunk.

    **Fact extraction is the agent's job, not the brain's.**

    Resolution order:
      1. `agent_facts` argument — pre-extracted facts the calling agent
         (OpenClaw / Hermes) already pulled out using its own LLM. This
         is the **preferred path** because it keeps the brain out of the
         LLM-loading business entirely. Pass them via `brain_remember(
         facts=[...])` or via `extract_facts_from_chunk(..., agent_facts=[...])`.
      2. Regex heuristics — lightweight, no model load. Catches obvious
         phrasing like "Duckets said X" / "Always do Y". Misses implied
         facts but costs zero VRAM.

    The brain NEVER auto-loads a chat model for fact extraction. If you
    want LLM-based extraction, the agent should pass the facts via
    `agent_facts=` (or `brain_remember(facts=[...])`). This avoids running
    a 4-9B chat model alongside the embedding + reranker models just for
    consolidation.

    `DUCKBOT_REGEX_ONLY=1` and `DUCKBOT_NO_LLM_EXTRACTION=1` (legacy)
    are accepted for back-compat but no longer change behavior — we
    don't load chat models regardless.

    The lower-level `extract_facts_via_llm()` is still exported for
    callers that DO want to drive an LLM explicitly (e.g. an offline
    batch job that already has the chat model loaded for other reasons).
    """
    seen_text: set[str] = set()
    facts: list[ExtractedFact] = []

    # Preferred: agent already extracted the facts
    if agent_facts:
        for text in agent_facts:
            text = (text or "").strip()
            if not text or len(text) < 5 or len(text) > 300:
                continue
            if text in seen_text:
                continue
            seen_text.add(text)
            facts.append(ExtractedFact(
                text=text,
                kind="fact",  # generic — agent didn't classify
                source_chunk_id=chunk_id,
                source_path=source_path,
                confidence=0.85,  # agent-extracted is high-confidence
            ))
        if facts:
            return facts

    # Fallback: lightweight regex
    return _extract_via_regex(chunk_text, chunk_id, source_path)


_MIN_CHUNK_FOR_LLM = 200


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
    """Extract durable facts from `chunk_text` via a caller-provided chat
    model (mem0-style prompts).

    This helper is intentionally explicit-only: if `model` is omitted, it
    returns [] and does not look up any consolidation-model environment
    variable. That keeps the main brain path free of a second model.

    Pattern source: mem0 paper + open-source implementation (Apache 2.0).
    """
    model_name = (model or "").strip()
    if not model_name:
        return []
    try:
        from .llm_client import chat_completion
    except ImportError:
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
            model=model_name,
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


def extract_facts_from_chunks(
    chunks: Iterable[dict],
    *,
    agent_facts_map: Optional[dict[str, list[str]]] = None,
) -> list[ExtractedFact]:
    """Bulk version. Chunks must have {id, text, metadata.source_path}.

    `agent_facts_map` is an optional {chunk_id: [fact, fact, ...]} dict
    holding pre-extracted facts per chunk — keeps extraction in the
    agent's hands (no brain-side LLM load).
    """
    all_facts: list[ExtractedFact] = []
    for chunk in chunks:
        cid = chunk["id"]
        facts = extract_facts_from_chunk(
            chunk["text"],
            cid,
            chunk.get("metadata", {}).get("source_path", "<unknown>"),
            agent_facts=(agent_facts_map or {}).get(cid),
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
    lead_tokens = {"duckets", "user", "he", "she", "they", "we", "i", "the"}

    def word_set(s: str) -> set[str]:
        words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'_-]*", s.lower())
        while words and words[0] in lead_tokens:
            words = words[1:]
        return set(words)

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
