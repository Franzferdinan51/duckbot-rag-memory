"""
experience.py — reusable-experience memory (Phase D, Hy "Working" scene).

Facts say what is true; episodes say HOW a task got done. An episode is
one PROCEDURAL-tier chunk, layer="experience", whose text is the full
record (task / context / steps / outcome) and whose metadata carries
the task line + outcome for filtering. Agents record episodes as they
finish work (MCP record_experience); scripts/mine_experience.py backfills
them from session summaries with the Hy-Memory writer. Successful
episodes are the raw material the skill-candidate pipeline promotes.

Import as src.experience.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

OUTCOMES = ("success", "failure", "partial", "unknown")


def _episode_text(task: str, context: str, steps: str, outcome: str) -> str:
    parts = [f"Task: {task}"]
    if context:
        parts.append(f"Context: {context}")
    parts.append(f"Steps: {steps}")
    parts.append(f"Outcome: {outcome}")
    return "\n".join(parts)


async def record_episode(mem, *, task: str, steps: str, context: str = "",
                         outcome: str = "unknown",
                         source_path: str = "<episode>"):
    """Store one experience episode. Returns the RememberResult."""
    if outcome not in OUTCOMES:
        outcome = "unknown"
    text = _episode_text(task.strip(), (context or "").strip(),
                         steps.strip(), outcome)
    return await mem.remember(
        text, source_path=source_path, force_tier="procedural",
        metadata={"layer": "experience", "task": task.strip()[:200],
                  "outcome": outcome})


async def recall_experience(mem, query: str, *, k: int = 5) -> list[dict]:
    """Recall episodes relevant to a task. Returns [{text, task, outcome}]."""
    results, _stats = await mem.recall(query, k=max(k * 3, 15),
                                       tier="procedural")
    out: list[dict] = []
    for r in results:
        meta = r.metadata or {}
        if meta.get("layer") != "experience":
            continue
        out.append({"text": r.text, "task": meta.get("task"),
                    "outcome": meta.get("outcome")})
        if len(out) >= k:
            break
    return out


async def mine_episodes(mem, text: str, *, model: Optional[str] = None,
                        source_path: str = "<mined>") -> list[dict]:
    """Writer-mine episodes from a text and store them. Fail-open -> []."""
    if not model or not text.strip():
        return []
    from . import hy_memory

    episodes = hy_memory.extract_episodes(text, model=model)
    stored: list[dict] = []
    for ep in episodes:
        await record_episode(mem, task=ep["task"], steps=ep["steps"],
                             context=ep["context"], outcome=ep["outcome"],
                             source_path=source_path)
        stored.append(ep)
    return stored
