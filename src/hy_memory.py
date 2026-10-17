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


# ---------------------------------------------------------------------------
# Phase 5: update semantics. Extraction alone appends; this pass lets the
# writer see the existing bank entries a chunk relates to and emit
# add / update / retire operations instead of piling up duplicates.
# ---------------------------------------------------------------------------

HY_MEMORY_UPDATE_PROMPT = """You maintain a long-term memory bank. Below are \
EXISTING memories related to a new piece of text, then the NEW text itself. \
Decide the operations that keep the bank accurate and free of duplicates: \
"add" for genuinely new facts, "update" when the new text supersedes an \
existing memory (give the existing memory's exact text as "target" and the \
corrected full statement as "text"), "retire" when an existing memory is \
now false and nothing replaces it (target only). Do not repeat facts the \
bank already states correctly. Return only a JSON object {"ops": [...]} \
with no other text. Each op is {"op": "add"|"update"|"retire", \
"target": string|null, "text": string|null}."""


def parse_memory_ops(text: Optional[str]) -> list[dict]:
    """Parse a Hy-Memory update reply into validated op dicts."""
    if not text:
        return []
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        data = json.loads(cleaned)
    except (ValueError, TypeError):
        return []
    if isinstance(data, dict):
        items = data.get("ops")
    elif isinstance(data, list):
        items = data
    else:
        return []
    if not isinstance(items, list):
        return []
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        op = item.get("op")
        if op not in ("add", "update", "retire"):
            continue
        target = item.get("target")
        new_text = item.get("text")
        entry = {
            "op": op,
            "target": target.strip() if isinstance(target, str) else None,
            "text": new_text.strip() if isinstance(new_text, str) else None,
        }
        if op == "add" and not entry["text"]:
            continue
        if op in ("update", "retire") and not entry["target"]:
            continue
        if op == "update" and not entry["text"]:
            continue
        out.append(entry)
    return out


def plan_memory_updates(
    chunk_text: str,
    existing: list[str],
    *,
    model: Optional[str] = None,
    max_tokens: int = 768,
    timeout: float = 120.0,
) -> list[dict]:
    """Ask Hy-Memory how a chunk should change the bank (add/update/retire).

    Explicit-only and silent-failure like extract_memories_hy: no model,
    no chunk, or any failure returns [].
    """
    model_name = (model or "").strip()
    if not model_name or not chunk_text.strip():
        return []
    from . import llm_client

    bank_block = "\n".join(f"- {e}" for e in existing) if existing else "(none)"
    user = f"EXISTING memories:\n{bank_block}\n\nNEW text:\n{chunk_text}"
    reply = llm_client.chat_completion(
        [
            {"role": "system", "content": HY_MEMORY_UPDATE_PROMPT},
            {"role": "user", "content": user},
        ],
        model=model_name,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return parse_memory_ops(reply)


# ---------------------------------------------------------------------------
# Phase A (Hy-Memory framework upgrade): conflict reconciliation.
#
# The write path's near-dup supersede (memory.py) only catches restatements
# (cosine distance < 0.08). A *changed* fact — different text, same topic —
# slips past it, leaving the stale fact current forever. reconcile() asks
# the writer model (RL-trained for exactly this: attributed conflict
# reconciliation) to classify each same-topic stored memory against the
# new one so the caller can supersede what the new fact updates or
# contradicts. Same discipline as the rest of this module: explicit model,
# silent failure (returns [] -> caller keeps everything, today's behavior).
# ---------------------------------------------------------------------------

HY_MEMORY_RECONCILE_PROMPT = """You are reconciling a NEW memory statement \
against EXISTING stored memories on the same topic. For each existing \
memory, classify its relation to the NEW memory:
- "same": it states the same fact (wording may differ).
- "updates": the NEW memory changes, corrects, or supersedes it; the \
existing memory is now out of date.
- "contradicts": the NEW memory conflicts with it; both cannot be true now.
- "unrelated": a different fact; both should be kept.
Return only a JSON object of the form {"verdicts": [{"index": 0, \
"relation": "updates", "evidence": "short reason"}]} with exactly one \
entry per existing memory, in the given order. No other text."""

_RELATIONS = ("same", "updates", "contradicts", "unrelated")


def parse_verdicts(text: Optional[str], n: int) -> list[dict]:
    """Parse a reconcile reply into validated verdict dicts.

    Strict: an entry survives only with an in-range integer index and a
    known relation. Anything malformed yields fewer (or zero) verdicts;
    the caller treats missing verdicts as "keep" (fail-open).
    """
    if not text or n <= 0:
        return []
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        data = json.loads(cleaned)
    except (ValueError, TypeError):
        return []
    if isinstance(data, dict):
        items = data.get("verdicts")
    elif isinstance(data, list):
        items = data
    else:
        return []
    if not isinstance(items, list):
        return []
    out: list[dict] = []
    seen: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        rel = item.get("relation")
        if not isinstance(idx, int) or isinstance(idx, bool):
            continue
        if idx < 0 or idx >= n or idx in seen:
            continue
        if rel not in _RELATIONS:
            continue
        seen.add(idx)
        ev = item.get("evidence")
        out.append({
            "index": idx,
            "relation": rel,
            "evidence": ev.strip()[:200] if isinstance(ev, str) else None,
        })
    return out


def reconcile(
    new_text: str,
    existing: list[str],
    *,
    model: Optional[str] = None,
    max_tokens: int = 512,
    timeout: float = 120.0,
) -> list[dict]:
    """Classify stored memories against a new fact via the writer model.

    Returns validated verdicts (see parse_verdicts). Explicit-only and
    silent-failure: no model, empty inputs, or any error returns [].
    """
    model_name = (model or "").strip()
    if not model_name or not new_text.strip() or not existing:
        return []
    from . import llm_client

    bank_block = "\n".join(f"[{i}] {e}" for i, e in enumerate(existing))
    user = f"EXISTING memories:\n{bank_block}\n\nNEW memory:\n{new_text}"
    reply = llm_client.chat_completion(
        [
            {"role": "system", "content": HY_MEMORY_RECONCILE_PROMPT},
            {"role": "user", "content": user},
        ],
        model=model_name,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return parse_verdicts(reply, len(existing))


# ---------------------------------------------------------------------------
# Phase B (Hy-Memory framework upgrade): schema induction + intentions.
#
# Hy's hierarchy tops the fact layer with SCHEMAS (patterns induced across
# many facts) and prospective INTENTIONS (plans/commitments as first-class
# memory). Both are writer-model passes, same discipline as the rest of
# this module: explicit model, strict parsing, silent failure -> [].
# ---------------------------------------------------------------------------

HY_MEMORY_SCHEMA_PROMPT = """Given the stored facts below about a person, \
induce up to 3 general SCHEMA statements: stable patterns, preferences, \
habits, or standing rules that MULTIPLE facts support. Each schema must \
be a concise, self-contained statement that generalizes the facts without \
inventing anything they do not support. If the facts support no pattern, \
return an empty list. Return only a JSON object with a "schemas" list of \
strings, with no other text."""

HY_MEMORY_INTENTION_PROMPT = """Extract prospective intentions from the \
text below: plans, commitments, scheduled actions, or things the person \
intends to do later. For each, write a concise self-contained statement, \
plus "due" (an ISO date YYYY-MM-DD if a time is stated, else null) and \
"trigger" (a short condition like "when the part arrives", else null). \
Do not extract facts about the past. If there are no intentions, return \
an empty list. Return only a JSON object of the form {"intentions": \
[{"text": "...", "due": null, "trigger": null}]}, with no other text."""


def _parse_string_list(text: Optional[str], key: str) -> list[str]:
    if not text:
        return []
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        data = json.loads(cleaned)
    except (ValueError, TypeError):
        return []
    if isinstance(data, dict):
        items = data.get(key)
    elif isinstance(data, list):
        items = data
    else:
        return []
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        if isinstance(item, str) and item.strip() and item.strip() not in out:
            out.append(item.strip())
    return out


def induce_schemas(
    facts: list[str],
    *,
    model: Optional[str] = None,
    max_tokens: int = 512,
    timeout: float = 120.0,
) -> list[str]:
    """Induce schema statements from a set of related facts."""
    model_name = (model or "").strip()
    if not model_name or len(facts) < 2:
        return []
    from . import llm_client

    fact_block = "\n".join(f"- {f}" for f in facts)
    reply = llm_client.chat_completion(
        [
            {"role": "system", "content": HY_MEMORY_SCHEMA_PROMPT},
            {"role": "user", "content": f"FACTS:\n{fact_block}"},
        ],
        model=model_name,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return _parse_string_list(reply, "schemas")


def parse_intentions(text: Optional[str]) -> list[dict]:
    """Parse an intentions reply: [{text, due, trigger}] strictly."""
    if not text:
        return []
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        data = json.loads(cleaned)
    except (ValueError, TypeError):
        return []
    if isinstance(data, dict):
        items = data.get("intentions")
    elif isinstance(data, list):
        items = data
    else:
        return []
    if not isinstance(items, list):
        return []
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        t = item.get("text")
        if not isinstance(t, str) or not t.strip():
            continue
        due = item.get("due")
        trigger = item.get("trigger")
        out.append({
            "text": t.strip(),
            "due": due.strip() if isinstance(due, str) and due.strip() else None,
            "trigger": trigger.strip() if isinstance(trigger, str) and trigger.strip() else None,
        })
    return out


def extract_intentions(
    text: str,
    *,
    model: Optional[str] = None,
    max_tokens: int = 512,
    timeout: float = 120.0,
) -> list[dict]:
    """Extract prospective intentions (plans/commitments) from text."""
    model_name = (model or "").strip()
    if not model_name or not text.strip():
        return []
    from . import llm_client

    reply = llm_client.chat_completion(
        [
            {"role": "system", "content": HY_MEMORY_INTENTION_PROMPT},
            {"role": "user", "content": text},
        ],
        model=model_name,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return parse_intentions(reply)


# ---------------------------------------------------------------------------
# Phase D (Hy-Memory framework upgrade): experience episodes.
#
# Hy's Working scene stores reusable EXPERIENCE alongside facts: how a
# task was actually accomplished, so an agent can recall the procedure
# next time instead of re-deriving it. extract_episodes mines episode
# records {task, context, steps, outcome} from session summaries / logs.
# ---------------------------------------------------------------------------

HY_MEMORY_EPISODE_PROMPT = """Extract reusable EXPERIENCE episodes from \
the text below: tasks that were actually carried out. For each, give \
"task" (what was being done), "context" (the situation/constraints), \
"steps" (the approach that was taken, as one concise string), and \
"outcome" (one of "success", "failure", "partial"). Only include tasks \
with a concrete approach worth reusing — not plans, not chit-chat. If \
there are none, return an empty list. Return only a JSON object of the \
form {"episodes": [{"task": "...", "context": "...", "steps": "...", \
"outcome": "success"}]}, with no other text."""

_OUTCOMES = ("success", "failure", "partial")


def parse_episodes(text: Optional[str]) -> list[dict]:
    """Parse an episodes reply into validated episode dicts."""
    if not text:
        return []
    cleaned = _FENCE_RE.sub("", text).strip()
    try:
        data = json.loads(cleaned)
    except (ValueError, TypeError):
        return []
    if isinstance(data, dict):
        items = data.get("episodes")
    elif isinstance(data, list):
        items = data
    else:
        return []
    if not isinstance(items, list):
        return []
    out: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        task = item.get("task")
        steps = item.get("steps")
        if not isinstance(task, str) or not task.strip():
            continue
        if not isinstance(steps, str) or not steps.strip():
            continue
        ctx = item.get("context")
        outcome = item.get("outcome")
        out.append({
            "task": task.strip(),
            "context": ctx.strip() if isinstance(ctx, str) and ctx.strip() else "",
            "steps": steps.strip(),
            "outcome": outcome if outcome in _OUTCOMES else "partial",
        })
    return out


def extract_episodes(
    text: str,
    *,
    model: Optional[str] = None,
    max_tokens: int = 1024,
    timeout: float = 120.0,
) -> list[dict]:
    """Mine reusable experience episodes from a session summary / log."""
    model_name = (model or "").strip()
    if not model_name or not text.strip():
        return []
    from . import llm_client

    reply = llm_client.chat_completion(
        [
            {"role": "system", "content": HY_MEMORY_EPISODE_PROMPT},
            {"role": "user", "content": text},
        ],
        model=model_name,
        temperature=0.0,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return parse_episodes(reply)
