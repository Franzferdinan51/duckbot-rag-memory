"""
upgrade_report.py — Hy-Memory upgrade acceptance metrics (Phase G).

Assembles the numbers that say whether the upgraded memory system is
healthy, on top of the existing eval harness (src/eval.py):

  census         chunk counts per collection and memory layer, plus
                 supersession totals and how many came from Phase A
                 reconciliation (superseded_via=hy-reconcile)
  graph          entity/relationship counts from the Phase C graph store
  golden         recall@5 / recall@10 / MRR over benchmarks/golden.jsonl
                 via eval.run_eval (optional; recall-only, no writes)

Import as src.upgrade_report.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_COLLECTIONS = ("semantic", "episodic", "procedural")


async def layer_census(mem) -> dict:
    """Count stored chunks by collection + layer + supersession state."""
    store, _ = await mem._ensure_initialized()
    census: dict = {"collections": {}, "layers": {}, "total": 0,
                    "superseded": 0, "superseded_via_reconcile": 0}
    for tier, coll in store.all_collections.items():
        name = tier.value if hasattr(tier, "value") else str(tier)
        try:
            data = coll.get(include=["metadatas"])
        except Exception:
            logger.exception("census failed for %s", name)
            continue
        metas = data.get("metadatas") or []
        census["collections"][name] = len(metas)
        census["total"] += len(metas)
        for meta in metas:
            meta = meta or {}
            layer = meta.get("layer") or "(base)"
            census["layers"][layer] = census["layers"].get(layer, 0) + 1
            if meta.get("superseded_by"):
                census["superseded"] += 1
                if meta.get("superseded_via") == "hy-reconcile":
                    census["superseded_via_reconcile"] += 1
    return census


def graph_stats(graph_path: Optional[str | Path] = None) -> dict:
    """Entity/relationship counts from the temporal graph store."""
    from . import graphwire
    from .graph import Graph
    path = Path(graph_path) if graph_path else graphwire.graph_path()
    if not Path(path).exists():
        return {"exists": False, "entities": 0, "relationships": 0}
    try:
        stats = Graph(path).stats()
    except Exception:
        logger.exception("graph stats failed")
        return {"exists": True, "error": True}
    return {"exists": True, **stats}


async def golden_eval(benchmark_path: str | Path, *, store=None,
                      embedder=None) -> dict:
    """Run the golden benchmark through the existing eval harness."""
    from dataclasses import asdict, is_dataclass
    from .eval import run_eval
    summary = await run_eval(benchmark_path, store=store, embedder=embedder)
    if hasattr(summary, "to_dict"):
        return summary.to_dict()
    return asdict(summary) if is_dataclass(summary) else dict(vars(summary))


async def build_report(mem, *, benchmark: Optional[str | Path] = None,
                       store=None, embedder=None,
                       graph_path: Optional[str | Path] = None) -> dict:
    report = {"census": await layer_census(mem), "graph": graph_stats(graph_path)}
    if benchmark is not None:
        report["golden"] = await golden_eval(benchmark, store=store,
                                             embedder=embedder)
    return report
