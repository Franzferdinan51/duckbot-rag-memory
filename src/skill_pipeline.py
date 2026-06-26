"""
skill_pipeline.py — agent-driven skill candidate → skill pipeline.

Storage-only. No LLM. The brain is a substrate; the agent is the author.

Flow:
  1. Agent finishes a task.
  2. Agent calls brain_remember(..., kind="skill_candidate") → brain stamps
     a chunk in the procedural tier with metadata.kind="skill_candidate".
     No LLM call. Returns chunk_id immediately (blocking, so the agent
     can promote it later).
  3. Later (end of session / consolidate pass), agent calls brain_skills_list
     to see candidates, picks which to promote, writes the SKILL.md itself
     using its own LLM context, then calls brain_skills_promote with the
     chunk_id + agent-authored content.
  4. The brain writes the SKILL.md (pure template via skillgen.write_skill)
     and marks the chunk as promoted=True.

Why agent-driven (not LLM-driven):
  The agent that did the task has the full context (what went wrong, what
  worked, what the user prefers). A separate LLM call from the brain trying
  to extract a SKILL.md from a chunk is at best a lossy summary, and at
  worst hallucinates. Also: the brain pays zero VRAM because it never
  spins up a generative LLM — only the embedding model runs.

Design constraint:
  Candidates go in the PROCEDURAL tier. Procedural memories persist longest
  (they're rules/procedures), which matches the intent: a candidate skill
  is a procedure the agent wants to re-use. This also keeps episodic clean.
"""

# MIT License — see LICENSE in the repository root.


from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from src.connectors.base import Brain, RememberResult, _run_async
from src.memory import Memory
from src.tier import Tier


# -----------------------------------------------------------------------------
# Stamp a candidate (no LLM — just store + embed)
# -----------------------------------------------------------------------------

def stamp_skill_candidate(
    text: str,
    source: str = "agent://skill-candidate",
    summary: str = "",
    importance: float = 0.6,
    trust_level: str = "full",
    brain: Optional[Brain] = None,
) -> RememberResult:
    """Stamp a skill-candidate chunk in the procedural tier.

    No LLM call. The brain just embeds + stores. The agent owns the
    decision of whether this is worth promoting to a full SKILL.md.

    Args:
        text: the memory content (what the agent did + key decisions).
        source: provenance tag.
        summary: optional short label for the candidate list. Defaults
            to a truncated form of the text.
        importance: 0..1 score used for ranking in suggest/list. Default
            0.6 — higher than baseline (0.5) because the agent thought
            it was worth remembering.
        trust_level: "full" (default) bypasses the injection scan —
            trust the agent since candidates are agent-authored. "standard"
            runs the scan and quarantines suspicious content. Use "standard"
            if the agent is being driven by user input you don't fully
            trust (e.g. an untrusted skill-generation script).
        brain: optional Brain facade override (used by tests).

    Returns:
        RememberResult with chunk_id. The caller should remember the
        chunk_id so it can promote the candidate later.
    """
    if brain is None:
        brain = Brain()

    if trust_level not in ("full", "standard"):
        # Don't raise — return a clear error via a sentinel? Use a
        # dummy remember() that returns an error chunk_id so the
        # caller can detect misuse without an exception.
        # Actually keep it simple: raise ValueError.
        raise ValueError(
            f"trust_level must be 'full' or 'standard', got {trust_level!r}"
        )

    metadata = {
        "kind": "skill_candidate",
        "promoted": False,
        "candidate_summary": (summary or text[:200]).strip(),
        "importance": importance,
        "trust_level": trust_level,
    }
    return brain.remember(
        text=text,
        source_path=source,
        metadata=metadata,
        force_tier="procedural",
        # trust_level=full: skip the scan (default — candidates are
        # agent-authored, scan would quarantine legitimate shell
        # commands and code snippets).
        # trust_level=standard: run the scan (treat as untrusted input
        # — e.g. agent reading skill suggestions from a user prompt).
        skip_scan=(trust_level == "full"),
    )


# -----------------------------------------------------------------------------
# List / suggest candidates
# -----------------------------------------------------------------------------

def list_candidates(
    brain: Optional[Brain] = None,
    include_promoted: bool = False,
    k: int = 50,
) -> list[dict]:
    """List skill-candidate chunks from the procedural tier.

    Returns a list of dicts sorted by recency (created_at desc):
        {
            chunk_id, text, summary, importance, promoted,
            promoted_skill_slug (if promoted), created_at, source_path
        }

    No LLM call. Pure metadata scan over the procedural collection.

    Args:
        brain: optional Brain facade override (unused for the query, kept
            for API symmetry with stamp/promote).
        include_promoted: if False (default), only return unpromoted
            candidates.
        k: cap on results. Must be > 0 (Chroma errors on limit <= 0).
    """
    if k <= 0:
        return {"error": "k must be a positive integer"}
    mem = Memory()
    store, _ = _run_async(mem._ensure_initialized())
    coll = store.collection_for(Tier.PROCEDURAL)

    # Chroma where-clause. We fetch all candidates and filter `promoted`
    # in Python for cross-version robustness (some Chroma versions don't
    # support $ne in compound where clauses).
    res = coll.get(
        where={"kind": "skill_candidate"},
        limit=k * 3 if not include_promoted else k,
        include=["documents", "metadatas"],
    )
    if not res or not res.get("ids"):
        return []

    out: list[dict] = []
    for i, cid in enumerate(res["ids"]):
        md = dict(res["metadatas"][i] or {})
        promoted = bool(md.get("promoted"))
        if promoted and not include_promoted:
            continue
        out.append({
            "chunk_id": cid,
            "text": res["documents"][i],
            "summary": md.get("candidate_summary", ""),
            "importance": float(md.get("importance", 0.5) or 0.5),
            "promoted": promoted,
            "promoted_skill_slug": md.get("promoted_skill_slug", ""),
            "promoted_at": md.get("promoted_at"),
            "created_at": float(md.get("created_at", 0.0) or 0.0),
            "source_path": md.get("source_path", md.get("source", "")),
        })

    # Sort by recency desc, then importance desc.
    out.sort(key=lambda c: (c["created_at"], c["importance"]), reverse=True)
    return out[:k]


def suggest_candidates(
    query: str,
    brain: Optional[Brain] = None,
    k: int = 5,
) -> list[dict]:
    """Semantic top-N skill candidates matching a query.

    Uses the existing hybrid retrieval (vector + BM25 + RRF) scoped to
    the procedural tier, then filters to unpromoted skill candidates.
    This is the "I just worked on X — are there candidate skills about
    X?" entry point.

    No LLM call. The query is embedded by the existing embedder and
    ranked by the existing retrieval stack.

    Args:
        query: semantic anchor (e.g. "docker container restart").
        brain: optional Brain facade override.
        k: max candidates to return.

    Returns:
        Same dict shape as list_candidates, plus a `score` field.
    """
    if brain is None:
        brain = Brain()

    # Over-fetch so we have enough after filtering.
    results = brain.recall(query=query, tier="procedural", k=k * 5)
    candidates: list[dict] = []
    for r in results:
        md = getattr(r, "metadata", None) or {}
        if md.get("kind") != "skill_candidate":
            continue
        if md.get("promoted"):
            continue
        candidates.append({
            "chunk_id": getattr(r, "chunk_id", ""),
            "text": getattr(r, "text", ""),
            "summary": md.get("candidate_summary", ""),
            "importance": float(md.get("importance", 0.5) or 0.5),
            "promoted": False,
            "created_at": float(md.get("created_at", 0.0) or 0.0),
            "source_path": md.get("source_path", md.get("source", "")),
            "score": float(getattr(r, "score", 0.0) or 0.0),
        })
        if len(candidates) >= k:
            break
    return candidates


# -----------------------------------------------------------------------------
# Promote a candidate to a full SKILL.md (pure template — no LLM)
# -----------------------------------------------------------------------------

def promote_candidate(
    chunk_id: str,
    name: str,
    description: str,
    instructions: Optional[list[str]] = None,
    brain: Optional[Brain] = None,
    example: str = "",
    emoji: Optional[str] = None,
    overwrite: bool = False,
    skills_dir: Optional[Path] = None,
    instructions_markdown: Optional[str] = None,
) -> dict:
    """Promote a skill candidate to a full SKILL.md file.

    The brain is pure storage + template here:
      1. Validate the candidate exists + is not already promoted.
      2. Write the SKILL.md (skillgen.write_skill — pure templating, no LLM).
      3. Mark the chunk as promoted=True + stamp promoted_at + slug.

    The agent authored the name/description/instructions using its own LLM
    context. The brain never calls an LLM.

    Args:
        chunk_id: the candidate to promote.
        name: human-readable skill name (agent-authored).
        description: one-line trigger (agent-authored).
        instructions: step-by-step instructions (agent-authored). Use
            this for simple flat lists. For richer SKILL.md bodies
            (headings, code blocks, tables), pass `instructions_markdown`
            instead — it overrides the flat list.
        instructions_markdown: optional rich markdown body. When set,
            used in place of `instructions` to render the SKILL.md
            body. Lets the agent author full markdown without going
            through a flat list.
        brain: unused (kept for API symmetry), the store is accessed
            via Memory() directly.
        example: optional worked example (agent-authored).
        emoji: optional emoji override.
        overwrite: if the SKILL.md already exists, replace it.
        skills_dir: override the skills directory (tests).

    Returns:
        {"path", "slug", "chunk_id", "promoted": True}
        or {"error": "..."} on failure.
    """
    from src.skillgen import write_skill, render_from_memory

    if not name or not description:
        return {"error": "name and description are required"}
    if not instructions:
        return {"error": "instructions list is required (at least one step)"}

    mem = Memory()
    store, _ = _run_async(mem._ensure_initialized())
    coll = store.collection_for(Tier.PROCEDURAL)

    # 1. Validate the candidate exists.
    cur = coll.get(ids=[chunk_id], include=["metadatas"])
    if not cur or not cur.get("ids") or chunk_id not in cur["ids"]:
        return {"error": f"chunk not found: {chunk_id}"}
    md = dict(cur["metadatas"][0] or {})

    if md.get("kind") != "skill_candidate":
        return {"error": f"chunk {chunk_id} is not a skill_candidate (kind={md.get('kind')})"}

    already_promoted = bool(md.get("promoted"))
    if already_promoted and not overwrite:
        return {
            "error": f"chunk {chunk_id} was already promoted to '{md.get('promoted_skill_slug')}'",
            "hint": "pass overwrite=true to re-promote",
        }

    # 2. Write the SKILL.md (pure template, no LLM).
    # If the chunk was previously promoted and overwrite=True, recover the
    # original human-readable title from the existing SKILL.md so the skill
    # identity (slug + title) is preserved across re-promotions. The slug
    # is deterministic from the title via _slugify, so the path stays the
    # same and the old file is overwritten in place (no orphan).
    effective_name = name
    if already_promoted:
        old_slug = md.get("promoted_skill_slug", "")
        if old_slug:
            existing_path = (
                (Path(__file__).resolve().parent.parent / "skills" / old_slug / "SKILL.md")
                if skills_dir is None
                else (Path(skills_dir) / old_slug / "SKILL.md")
            )
            if existing_path.exists():
                try:
                    text = existing_path.read_text(encoding="utf-8")
                    parts = text.split("---", 2)
                    if len(parts) >= 3:
                        for line in parts[2].splitlines():
                            if line.startswith("# "):
                                recovered = line[2:].strip()
                                if recovered:
                                    effective_name = recovered
                                break
                except Exception:
                    pass

    body = render_from_memory(
        name=effective_name,
        description=description,
        instructions=instructions or [],
        example=example,
        emoji=emoji,
    )

    # If the agent provided instructions_markdown, replace the rendered
    # body with the rich markdown content. The flat instructions list is
    # still available as fallback in the body if markdown is empty.
    if instructions_markdown is not None and instructions_markdown.strip():
        # Keep the title (# ...) that render_from_memory added; replace
        # the body content (after the title + the "When to Use" / "Instructions"
        # sections) with the agent-authored markdown.
        lines = body.splitlines()
        title_idx = next(
            (i for i, ln in enumerate(lines) if ln.startswith("# ")),
            None,
        )
        if title_idx is not None:
            # Find end of the Instructions section to splice in markdown
            new_body = lines[:title_idx + 1] + [""] + instructions_markdown.strip().splitlines() + [""]
            if example:
                new_body += ["## Example", "", example.strip(), ""]
            body = "\n".join(new_body)
    if skills_dir is None:
        repo_root = Path(__file__).resolve().parent.parent
        skills_dir = repo_root / "skills"
    try:
        path = write_skill(
            skills_dir=skills_dir,
            name=effective_name,
            description=description,
            body_markdown=body,
            emoji=emoji,
            overwrite=overwrite,
        )
    except FileExistsError as e:
        return {"error": str(e), "hint": "pass overwrite=true to replace"}

    # 3. Mark the chunk as promoted.
    md["promoted"] = True
    md["promoted_at"] = time.time()
    md["promoted_skill_slug"] = path.parent.name
    coll.update(ids=[chunk_id], metadatas=[md])

    return {
        "path": str(path),
        "slug": path.parent.name,
        "chunk_id": chunk_id,
        "promoted": True,
        "previously_promoted": already_promoted,
    }


# -----------------------------------------------------------------------------
# Stats helper (for the skill pipeline surface)
# -----------------------------------------------------------------------------

def candidate_stats(brain: Optional[Brain] = None) -> dict:
    """Quick counts: total candidates, unpromoted, promoted.

    No LLM call. One metadata scan over the procedural tier.
    """
    mem = Memory()
    store, _ = _run_async(mem._ensure_initialized())
    coll = store.collection_for(Tier.PROCEDURAL)
    res = coll.get(where={"kind": "skill_candidate"}, include=["metadatas"])
    if not res or not res.get("ids"):
        return {"total": 0, "promoted": 0, "unpromoted": 0}
    promoted = sum(1 for m in res["metadatas"] if (m or {}).get("promoted"))
    total = len(res["ids"])
    return {"total": total, "promoted": promoted, "unpromoted": total - promoted}


__all__ = [
    "stamp_skill_candidate",
    "list_candidates",
    "suggest_candidates",
    "promote_candidate",
    "candidate_stats",
]
