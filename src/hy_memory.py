"""
hy_memory.py — optional Hy-Memory writer adapter for the consolidation path.

Hy-Memory (tencent/Hy-Memory-30B-A3B, Apache-2.0) is a specialist
long-term-memory writer model: given a chunk of conversation or notes, it
decides what is worth retaining and returns concise, self-contained
memory statements as JSON. This module is a thin adapter that runs that
extraction through the existing llm_client seam.

Design constraints (same philosophy as consolidate.py / llm_client.py):
- Explicit-only: the caller passes `model=` (the LM Studio id of a loaded
  Hy-Memory build). There is no env lookup and no default model. The
  brain never auto-loads a chat model; a batch job that already loaded
  Hy-Memory for other reasons may pass it here.
- Silent failure: transport errors, non-200 replies, and malformed JSON
  all return [] so the caller can fall back to regex extraction.
- Decoding follows Tencent's quickstart: temperature 0.0, no-think mode
  (the model's chat template defaults reasoning_effort to no_think).

Import as src.hy_memory (the watcher runs as `python -m src.watcher`).
"""
from __future__ import annotations

import json
import re
from typing import Optional

HY_MEMORY_EXTRACTION_PROMPT = """Extract explicit facts, preferences, \
decisions, and commitments from the text below. Write each memory as a \
concise, self-contained statement that makes sense on its own, without \
pronouns that depend on the surrounding text. Do not infer or invent \
information that is not stated. If nothing is worth remembering, return \
an empty list. Return only a JSON object with a "memories" list of \
strings, with no other text."""

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_memories(text: Optional[str]) -> list[str]:
    """Parse a Hy-Memory reply into a clean list of memory statements.

    Accepts the documented {"memories": [...]} object, a bare JSON list,
    or either wrapped in a markdown code fence. Returns [] on anything
    unparseable. Dedupes while preserving order.
    """
    if not text:
        return []
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        data = json.loads(cleaned)
    except (ValueError, TypeError):
        return []
    if isinstance(data, dict):
        items = data.get("memories")
    elif isinstance(data, list):
        items = data
    else:
        return []
    if not isinstance(items, list):
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def extract_memories_hy(
    chunk_text: str,
    *,
    model: Optional[str] = None,
    max_tokens: int = 512,
    timeout: float = 120.0,
) -> list[str]:
    """Extract memory statements from one chunk via a loaded Hy-Memory model.

    Returns [] when no model is given, the model is unreachable, or the
    reply cannot be parsed — the caller is expected to fall back to
    regex extraction, exactly as with extract_facts_via_llm.
    """
    model_name = (model or "").strip()
    if not model_name or not chunk_text.strip():
        return []
    from . import llm_client

    reply = llm_client.chat_completion(
        [
            {"role": "system", "content": HY_MEMORY_EXTRACTION_PROMPT},
            {"role": "user", "content": chunk_text},
        ],
        model=model_name,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return parse_memories(reply)
