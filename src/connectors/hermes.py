"""
connectors/hermes.py — Hermes agent integration for the DuckBot brain.

Hermes is a Telegram-polling bot, not an HTTP service (per MEMORY.md 2026-06-23).
So integration happens two ways:

  1. Python import (preferred when Hermes is in the same venv or has the
     duckbot-rag-memory repo on its sys.path):

         from duckbot_rag_memory.connectors.hermes import HermesBrain
         brain = HermesBrain()
         brain.remember("Duckets rotated the bot token today")
         results = brain.recall("bot token rotation", k=5)

  2. CLI shim (for Hermes agents that shell out):

         python -m src.cli hermes remember "Duckets rotated the bot token"
         python -m src.cli hermes recall "bot token rotation" 5
         python -m src.cli hermes block-read user
         python -m src.cli hermes stats
         python -m src.cli hermes scan "Ignore previous instructions"

Why both? Python imports are cleaner but require sys.path setup. The CLI
shim works from anywhere with no setup. Hermes is a bot framework, so the
operator can pick whichever is more convenient.

No HTTP, no sockets, no paid services. Local Python + SQLite.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Optional

from .base import Brain


# -----------------------------------------------------------------------------
# Module-level singleton (Hermes-style: "just import and use")
# -----------------------------------------------------------------------------

_DEFAULT_BRAIN: Optional[Brain] = None


def get_brain() -> Brain:
    """Lazy singleton. Hermes agents can call this without managing state."""
    global _DEFAULT_BRAIN
    if _DEFAULT_BRAIN is None:
        _DEFAULT_BRAIN = Brain()
    return _DEFAULT_BRAIN


# Backwards-compat: some Hermes scripts may import this name
HermesBrain = Brain


# -----------------------------------------------------------------------------
# Convenience functions (one-liners for Hermes agent scripts)
# -----------------------------------------------------------------------------

def remember(text: str, source: str = "<hermes>", **kwargs) -> dict:
    """Save a memory. Returns a dict with chunk_id, tier, quarantined, etc."""
    r = get_brain().remember(text, source_path=source, **kwargs)
    return r.to_dict()


def recall(query: str, k: int = 5, **kwargs) -> list[dict]:
    """Search memory. Returns a list of result dicts (already dict, not dataclass)."""
    results = get_brain().recall(query, k=k, **kwargs)
    return [r.to_dict() for r in results]


def reflect(**kwargs) -> dict:
    """Sleep-time consolidation. Returns dict with consolidation stats."""
    from src.memory import Memory
    import asyncio
    return asyncio.run(Memory().reflect(**kwargs))


def stats() -> dict:
    """One-glance brain snapshot."""
    return get_brain().stats().to_dict()


# Block helpers
def block_read(name: str) -> Optional[dict]:
    return get_brain().block_read(name)


def block_write(name: str, text: str) -> dict:
    return get_brain().block_write(name, text)


def block_append(name: str, text: str) -> dict:
    return get_brain().block_append(name, text)


def block_list() -> list[dict]:
    return get_brain().block_list()


# Graph helpers
def graph_upsert(name: str, kind: str = "concept") -> dict:
    return get_brain().graph_upsert_entity(name, kind)


def graph_relate(source: str, target: str, label: str) -> dict:
    return get_brain().graph_add_relationship(source, target, label)


def graph_query(name: Optional[str] = None, kind: Optional[str] = None) -> list[dict]:
    return get_brain().graph_query(name=name, kind=kind)


def graph_relationships(entity: str) -> list[dict]:
    return get_brain().graph_relationships(entity_name=entity)


# Injection / quarantine helpers
def scan(text: str) -> dict:
    """One-shot injection scan. Returns the scan summary (no quarantine)."""
    return get_brain().injection_scan(text)


def quarantine_list(status: str = "pending") -> list[dict]:
    return get_brain().quarantine_list(status=status)


def quarantine_review(scan_id: str, decision: str, reviewer: str = "hermes") -> dict:
    return get_brain().quarantine_review(scan_id, decision, reviewer=reviewer)


# -----------------------------------------------------------------------------
# CLI shim (for Hermes agents that shell out)
# -----------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    """
    `python -m src.cli hermes <verb> [args...]`

    Verbs:
      remember <text>             - save a memory
      recall <query> [k]          - search memory
      reflect [lookback_days]     - consolidate
      stats                       - one-glance brain snapshot
      scan <text>                 - one-shot injection scan
      block-read <name>           - read a memory block
      block-write <name> <text>   - write a memory block
      block-append <name> <text>  - append to a memory block
      block-list                  - list all memory blocks
      graph-upsert <name> [kind]  - add/update graph entity
      graph-relate <src> <tgt> <label>  - add graph relationship
      graph-query [name] [kind]   - query graph entities
      graph-relationships <name>  - get entity's active relationships
      quarantine-list [status]    - list quarantined (pending|approved|rejected|all)
      quarantine-review <id> <decision>  - approve|reject a quarantined chunk

    All output is JSON to stdout, so Hermes can parse it back.
    """
    argv = argv or sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(main.__doc__)
        return 0
    verb = argv[0]
    rest = argv[1:]

    out: Any
    try:
        if verb == "remember":
            text = " ".join(rest)
            if not text:
                print(json.dumps({"error": "remember requires <text>"}))
                return 1
            out = remember(text)
        elif verb == "recall":
            if not rest:
                print(json.dumps({"error": "recall requires <query>"}))
                return 1
            query = rest[0]
            k = int(rest[1]) if len(rest) > 1 else 5
            out = recall(query, k=k)
        elif verb == "reflect":
            lookback = int(rest[0]) if rest else 7
            out = reflect(lookback_days=lookback)
        elif verb == "stats":
            out = stats()
        elif verb == "scan":
            text = " ".join(rest)
            if not text:
                print(json.dumps({"error": "scan requires <text>"}))
                return 1
            out = scan(text)
        elif verb == "block-read":
            if not rest:
                print(json.dumps({"error": "block-read requires <name>"}))
                return 1
            out = block_read(rest[0])
        elif verb == "block-write":
            if len(rest) < 2:
                print(json.dumps({"error": "block-write requires <name> <text>"}))
                return 1
            out = block_write(rest[0], " ".join(rest[1:]))
        elif verb == "block-append":
            if len(rest) < 2:
                print(json.dumps({"error": "block-append requires <name> <text>"}))
                return 1
            out = block_append(rest[0], " ".join(rest[1:]))
        elif verb == "block-list":
            out = block_list()
        elif verb == "graph-upsert":
            if not rest:
                print(json.dumps({"error": "graph-upsert requires <name>"}))
                return 1
            kind = rest[1] if len(rest) > 1 else "concept"
            out = graph_upsert(rest[0], kind)
        elif verb == "graph-relate":
            if len(rest) < 3:
                print(json.dumps({"error": "graph-relate requires <source> <target> <label>"}))
                return 1
            out = graph_relate(rest[0], rest[1], rest[2])
        elif verb == "graph-query":
            name = rest[0] if len(rest) > 0 else None
            kind = rest[1] if len(rest) > 1 else None
            out = graph_query(name, kind)
        elif verb == "graph-relationships":
            if not rest:
                print(json.dumps({"error": "graph-relationships requires <entity>"}))
                return 1
            out = graph_relationships(rest[0])
        elif verb == "quarantine-list":
            status = rest[0] if rest else "pending"
            out = quarantine_list(status)
        elif verb == "quarantine-review":
            if len(rest) < 2:
                print(json.dumps({"error": "quarantine-review requires <scan_id> <decision>"}))
                return 1
            out = quarantine_review(rest[0], rest[1])
        else:
            print(json.dumps({"error": f"unknown verb: {verb}"}))
            return 1
        print(json.dumps(out, indent=2, default=str))
        return 0
    except Exception as e:
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
