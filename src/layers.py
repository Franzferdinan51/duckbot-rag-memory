"""
layers.py — the upper layers of the Hy-Memory hierarchy (Phase B).

The bank's ground floor is facts (semantic tier). Hy's architecture adds
two layers above it, both produced by writer-model passes in
src/hy_memory.py and stored back into the same store with `layer`
metadata, so every existing consumer keeps working:

  schema     — patterns induced across many facts (preferences, habits,
               standing rules). Stored in the PROCEDURAL tier,
               layer="schema", with derived_from pointing at fact ids.
  intention  — prospective memory: plans and commitments extracted from
               text, stored in the SEMANTIC tier, layer="intention",
               status="open", with optional due / trigger metadata.
               Intentions are things that should resurface at the right
               time, not facts about the past.

recall_context() assembles a layered answer for a query: current facts,
the schemas that bear on them, and open intentions — the shape an agent
actually wants at the start of a task.

All passes are explicit-model and fail-open (hy_memory discipline): a
missing model or a bad reply stores nothing and returns [].
Import as src.layers.
"""
from __future__ import annotations

import logging
from typing import Optional

from .tier import Tier

logger = logging.getLogger(__name__)


async def induce_schema_layer(mem, *, model: Optional[str] = None,
                              min_facts: int = 3, limit: int = 60) -> list[str]:
    """Induce schemas from the bank's current semantic facts and store them.

    Returns the schema statements stored ([] when the bank is too small
    or the model pass produced nothing).
    """
    if not model:
        return []
    from . import hy_memory

    coll = mem._store.collection_for(Tier.SEMANTIC)
    got = coll.get(include=["documents", "metadatas"])
    facts: list[str] = []
    fact_ids: list[str] = []
    ids = got.get("ids") or []
    for i, doc in enumerate(got.get("documents") or []):
        meta = (got.get("metadatas") or [[]])[i] if i < len(got.get("metadatas") or []) else {}
        meta = meta or {}
        if meta.get("superseded_by") or meta.get("layer"):
            continue
        if doc and len(doc.strip()) >= 10:
            facts.append(doc.strip())
            if i < len(ids):
                fact_ids.append(ids[i])
    if len(facts) < min_facts:
        return []
    schemas = hy_memory.induce_schemas(facts[:limit], model=model)
    stored: list[str] = []
    for s in schemas:
        await mem.remember(
            s, source_path="<schema-induction>", force_tier="procedural",
            metadata={"layer": "schema",
                      "derived_from": ",".join(fact_ids[:20])})
        stored.append(s)
    return stored


async def extract_intention_layer(mem, texts: list[str], *,
                                  model: Optional[str] = None,
                                  source_path: str = "<intentions>") -> list[dict]:
    """Extract prospective intentions from source texts and store them open."""
    if not model:
        return []
    from . import hy_memory

    out: list[dict] = []
    for t in texts:
        for intent in hy_memory.extract_intentions(t, model=model):
            meta = {"layer": "intention", "status": "open"}
            if intent.get("due"):
                meta["due"] = intent["due"]
            if intent.get("trigger"):
                meta["trigger"] = intent["trigger"]
            await mem.remember(intent["text"], source_path=source_path,
                               force_tier="semantic", metadata=meta)
            out.append(intent)
    return out


async def recall_context(mem, query: str, *, k: int = 5) -> dict:
    """Layer-aware recall: {facts, schemas, intentions} for a query.

    Facts are current (recall already filters superseded); intentions are
    open ones only. Each item is {text, tier} (+due/trigger on intentions).
    """
    results, _stats = await mem.recall(query, k=max(k * 3, 15))
    facts: list[dict] = []
    schemas: list[dict] = []
    intentions: list[dict] = []
    for r in results:
        meta = r.metadata or {}
        layer = meta.get("layer")
        if layer == "schema":
            schemas.append({"text": r.text, "tier": r.tier})
        elif layer == "intention":
            if meta.get("status", "open") == "open":
                intentions.append({"text": r.text, "tier": r.tier,
                                   "due": meta.get("due"),
                                   "trigger": meta.get("trigger")})
        else:
            facts.append({"text": r.text, "tier": r.tier})
    return {"facts": facts[:k], "schemas": schemas[:k],
            "intentions": intentions[:k]}
