"""connectors/openclaw_shim.py — OpenClaw CLI shim.

Parallel to `connectors/hermes.py`'s CLI shim, but delegates to the
shared 11-tool core agent surface (`src.extensions.tools`) so the
shell path matches what an OpenClaw agent sees via the JSON-RPC
adapter.

Why a separate shim (not just adding verbs to hermes.py)?

  - The Hermes shim speaks the Hermes contract (sync_turn, blocks,
    graph, quarantine). The OpenClaw shim speaks the OpenClaw
    contract (the 11 core tools + a generic `call <tool> '<json>'`
    escape hatch). Mixing them would confuse either side.
  - The OpenClaw shim is the recommended shell entry point for
    agents that want to test the JSON-RPC surface without standing
    up an OpenClaw gateway. Output is JSON so scripts can parse it.

Usage:

    python -m src.cli openclaw <verb> [args...]

Verbs:

    wake-up [--query Q] [-k K]               - one-call session-start context load
    recall <query> [-k K]                    - hybrid retrieval
    recall-verbatim <query>                  - original source bytes
    remember <text>                          - save a memory (non-blocking)
    remember-skill-candidate <text>          - stamp a skill candidate (no LLM)
    reflect [lookback_days]                  - consolidate episodic → semantic
    stats                                    - one-glance brain snapshot
    fsrs-review [tier] [-k K]                - spaced-repetition review queue
    decay-status [tier] [-k K]               - retention scoring for recent chunks
    search-verbatim <needle>                 - exact substring match
    skills-list                              - list unpromoted skill candidates
    skills-promote <chunk_id> <name>         - promote a candidate to SKILL.md
    tools                                    - list the 11 core tools
    call <tool> '<json-args>'                - generic dispatch (full surface)

All output is JSON to stdout, so OpenClaw cron jobs / shell scripts
can parse it back.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Optional

from src.extensions import tools as _surface


# -----------------------------------------------------------------------------
# Verb handlers — each maps to one of the 9 core tools
# -----------------------------------------------------------------------------


def _cmd_wake_up(rest: list[str]) -> Any:
    args: dict = {}
    i = 0
    while i < len(rest):
        if rest[i] in ("--query", "-q") and i + 1 < len(rest):
            args["query"] = rest[i + 1]
            i += 2
        elif rest[i] in ("-k", "--k") and i + 1 < len(rest):
            args["k"] = int(rest[i + 1])
            i += 2
        else:
            i += 1
    return _surface.dispatch("brain_wake_up", args)


def _cmd_recall(rest: list[str]) -> Any:
    if not rest:
        return {"error": "recall requires <query>"}
    # Strip flag-style args before joining the query.
    query_parts: list[str] = []
    k: Optional[int] = None
    i = 0
    while i < len(rest):
        if rest[i] in ("-k", "--k") and i + 1 < len(rest):
            k = int(rest[i + 1])
            i += 2
        else:
            query_parts.append(rest[i])
            i += 1
    if not query_parts:
        return {"error": "recall requires <query>"}
    args = {"query": " ".join(query_parts)}
    if k is not None:
        args["k"] = k
    return _surface.dispatch("brain_recall", args)


def _cmd_recall_verbatim(rest: list[str]) -> Any:
    if not rest:
        return {"error": "recall-verbatim requires <query>"}
    return _surface.dispatch("brain_recall_verbatim", {"query": " ".join(rest)})


def _cmd_remember(rest: list[str]) -> Any:
    text = " ".join(rest)
    if not text:
        return {"error": "remember requires <text>"}
    return _surface.dispatch("brain_remember", {"text": text, "source": "openclaw-cli://ad-hoc"})


def _cmd_remember_skill_candidate(rest: list[str]) -> Any:
    """Stamp a skill candidate. Convenience verb for the agent-driven pipeline."""
    text = " ".join(rest)
    if not text:
        return {"error": "remember-skill-candidate requires <text>"}
    return _surface.dispatch("brain_remember", {
        "text": text,
        "source": "openclaw-cli://skill-candidate",
        "kind": "skill_candidate",
    })


def _cmd_reflect(rest: list[str]) -> Any:
    args: dict = {}
    if rest:
        try:
            args["lookback_days"] = int(rest[0])
        except ValueError:
            pass
    return _surface.dispatch("brain_reflect", args)


def _cmd_stats(rest: list[str]) -> Any:
    return _surface.dispatch("brain_stats", {})


def _cmd_fsrs_review(rest: list[str]) -> Any:
    args: dict = {}
    if rest:
        # First positional could be a tier or a k.
        if rest[0] in ("working", "episodic", "semantic", "procedural"):
            args["tier"] = rest[0]
    # Look for -k anywhere.
    if "-k" in rest:
        idx = rest.index("-k")
        if idx + 1 < len(rest):
            args["k"] = int(rest[idx + 1])
    return _surface.dispatch("brain_fsrs_review", args)


def _cmd_decay_status(rest: list[str]) -> Any:
    args: dict = {}
    if rest:
        if rest[0] in ("working", "episodic", "semantic", "procedural"):
            args["tier"] = rest[0]
    if "-k" in rest:
        idx = rest.index("-k")
        if idx + 1 < len(rest):
            args["k"] = int(rest[idx + 1])
    return _surface.dispatch("brain_decay_status", args)


def _cmd_search_verbatim(rest: list[str]) -> Any:
    needle = " ".join(rest)
    if not needle:
        return {"error": "search-verbatim requires <needle>"}
    return _surface.dispatch("brain_search_verbatim", {"needle": needle})


def _cmd_skills_list(rest: list[str]) -> Any:
    """List unpromoted skill candidates (agent-driven pipeline)."""
    args: dict = {}
    if "--include-promoted" in rest:
        args["include_promoted"] = True
    if "-k" in rest:
        idx = rest.index("-k")
        if idx + 1 < len(rest):
            try:
                args["k"] = int(rest[idx + 1])
            except ValueError:
                pass
    return _surface.dispatch("brain_skills_list", args)


def _cmd_skills_suggest(rest: list[str]) -> Any:
    """Semantic top-N skill candidates matching a query.

    Usage: skills-suggest <query>... [-k N]
    """
    if not rest:
        return {"error": "skills-suggest requires <query>"}
    # Rejoin all args as the query (allows multi-word queries without quoting)
    args: dict = {"query": " ".join(rest)}
    if "-k" in rest:
        idx = rest.index("-k")
        if idx + 1 < len(rest):
            try:
                args["k"] = int(rest[idx + 1])
            except ValueError:
                pass
    return _surface.dispatch("brain_skills_suggest", args)


def _cmd_skills_promote(rest: list[str]) -> Any:
    """Promote a skill candidate to a full SKILL.md.

    Usage: skills-promote <chunk_id> <name> <description> <instruction1> [instruction2...]
    Or:    skills-promote <chunk_id> --json '<json-args>'

    The first form is shell-friendly; the second is for scripted use.
    """
    if len(rest) < 2:
        return {"error": "skills-promote requires <chunk_id> and skill content (name + description + instructions)"}

    # JSON form: skills-promote <chunk_id> --json '<json-args>'
    if "--json" in rest:
        idx = rest.index("--json")
        chunk_id = rest[0] if idx > 0 else ""
        raw = " ".join(rest[idx + 1:]).strip()
        if not chunk_id or not raw:
            return {"error": "skills-promote --json requires <chunk_id> '<json-args>'"}
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                return {"error": "json-args must be an object"}
        except json.JSONDecodeError as e:
            return {"error": f"json-args not valid JSON: {e}"}
        parsed["chunk_id"] = chunk_id
        return _surface.dispatch("brain_skills_promote", parsed)

    # Positional form: skills-promote <chunk_id> <name> <description> <instr1> [instr2 ...]
    chunk_id = rest[0]
    if len(rest) < 4:
        return {"error": "positional form requires: skills-promote <chunk_id> <name> <description> <instr1> [instr2 ...]"}
    name = rest[1]
    description = rest[2]
    instructions = rest[3:]
    return _surface.dispatch("brain_skills_promote", {
        "chunk_id": chunk_id,
        "name": name,
        "description": description,
        "instructions": instructions,
    })


def _cmd_tools(rest: list[str]) -> Any:
    """List the 11 core tools (same surface the JSON-RPC adapter exposes)."""
    return {"tools": _surface.tool_schemas(), "summary": _surface.summary()}


def _cmd_call(rest: list[str]) -> Any:
    """Generic dispatch: `call <tool> '<json-args>'`.

    The args string is parsed as JSON. Empty string = no args.
    """
    if not rest:
        return {"error": "call requires <tool> [json-args]"}
    tool = rest[0]
    raw_args = " ".join(rest[1:]).strip()
    args: dict = {}
    if raw_args:
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                args = parsed
            else:
                return {"error": f"call args must be a JSON object, got {type(parsed).__name__}"}
        except json.JSONDecodeError as e:
            return {"error": f"call args not valid JSON: {e}"}
    return _surface.dispatch(tool, args)


_VERBS: dict[str, Any] = {
    "wake-up": _cmd_wake_up,
    "wake_up": _cmd_wake_up,
    "recall": _cmd_recall,
    "recall-verbatim": _cmd_recall_verbatim,
    "recall_verbatim": _cmd_recall_verbatim,
    "remember": _cmd_remember,
    "remember-skill-candidate": _cmd_remember_skill_candidate,
    "remember_skill_candidate": _cmd_remember_skill_candidate,
    "reflect": _cmd_reflect,
    "stats": _cmd_stats,
    "fsrs-review": _cmd_fsrs_review,
    "fsrs_review": _cmd_fsrs_review,
    "decay-status": _cmd_decay_status,
    "decay_status": _cmd_decay_status,
    "search-verbatim": _cmd_search_verbatim,
    "search_verbatim": _cmd_search_verbatim,
    "skills-list": _cmd_skills_list,
    "skills_list": _cmd_skills_list,
    "skills-suggest": _cmd_skills_suggest,
    "skills_suggest": _cmd_skills_suggest,
    "skills-promote": _cmd_skills_promote,
    "skills_promote": _cmd_skills_promote,
    "tools": _cmd_tools,
    "call": _cmd_call,
}


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    """`python -m src.cli openclaw <verb> [args...]`"""
    argv = argv if argv is not None else sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(main.__doc__)
        return 0
    verb = argv[0]
    rest = argv[1:]

    handler = _VERBS.get(verb)
    if handler is None:
        print(json.dumps({
            "error": f"unknown verb: {verb}",
            "available": sorted(_VERBS.keys()),
        }))
        return 1

    try:
        out = handler(rest)
        print(json.dumps(out, indent=2, default=str))
        # Non-zero exit if the dispatch surfaced an error, so shell
        # pipelines / cron jobs can detect failure without parsing JSON.
        if isinstance(out, dict) and out.get("error"):
            return 2
        return 0
    except Exception as e:  # noqa: BLE001
        print(json.dumps({"error": str(e), "verb": verb}))
        return 1
