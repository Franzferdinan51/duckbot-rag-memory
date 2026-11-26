#!/usr/bin/env python3
"""Backfill / incrementally feed the temporal knowledge graph from the bank.

Why this exists: the graph feed inside Memory.remember() is opt-in
(DUCKBOT_GRAPH_FEED), and it was never enabled in the live deployment, so
after months of writes the graph held exactly 1 entity against ~29k
chunks and every graph feature (recall augmentation, brain_graph tools)
was silently working on an empty graph.

This script walks every tier collection, runs the regex EntityExtractor
over each live (non-superseded) chunk, and cognifies the results into
data/graph.db. Extraction is pure regex — no model load, safe to run any
time. cognify() is idempotent, and a watermark file
(data/graph_backfill_state.json) records processed chunk ids so nightly
incremental runs only touch new chunks.

Usage:
    .venv/bin/python scripts/backfill_graph.py            # incremental
    .venv/bin/python scripts/backfill_graph.py --full     # ignore watermark
    .venv/bin/python scripts/backfill_graph.py --dry-run  # count only
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src._env import load_repo_dotenv

load_repo_dotenv()

from src.entities import EntityExtractor  # noqa: E402
from src.graph import Graph  # noqa: E402
from src.graphwire import graph_path  # noqa: E402
from src.store import MemoryStore  # noqa: E402
from src.tier import Tier  # noqa: E402

STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "graph_backfill_state.json"
PAGE = 1000
BATCH = 250


def _load_state() -> set[str]:
    try:
        return set(json.loads(STATE_PATH.read_text()).get("done_ids", []))
    except Exception:
        return set()


def _save_state(done: set[str]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"done_ids": sorted(done), "updated_at": time.time()}))
    tmp.replace(STATE_PATH)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--full", action="store_true", help="ignore the watermark")
    ap.add_argument("--dry-run", action="store_true", help="scan and count, write nothing")
    args = ap.parse_args()

    done = set() if args.full else _load_state()
    store = MemoryStore()
    graph = None if args.dry_run else Graph(str(graph_path()))
    extractor = EntityExtractor()

    scanned = skipped = new = 0
    ents_added = trips_added = 0
    t0 = time.time()

    for tier in Tier:
        try:
            coll = store.collection_for(tier)
        except Exception:
            continue
        offset = 0
        while True:
            got = coll.get(include=["documents", "metadatas"], limit=PAGE, offset=offset)
            ids = got.get("ids") or []
            if not ids:
                break
            docs = got.get("documents") or []
            metas = got.get("metadatas") or []
            batch_texts: list[str] = []
            batch_sources: list[str] = []
            batch_ids: list[str] = []
            for i, cid in enumerate(ids):
                scanned += 1
                meta = metas[i] if i < len(metas) else None
                meta = meta or {}
                doc = docs[i] if i < len(docs) else ""
                if cid in done or meta.get("superseded_by") or not doc or len(doc.strip()) < 10:
                    skipped += 1
                    if cid not in done and not args.dry_run:
                        done.add(cid)  # never reconsider dead chunks
                    continue
                batch_texts.append(doc)
                batch_sources.append(str(meta.get("source_path") or tier.value))
                batch_ids.append(cid)
                if len(batch_texts) >= BATCH:
                    entities, triples = extractor.extract_batch(batch_texts, batch_sources)
                    if graph is not None:
                        res = extractor.cognify(graph, entities, triples, source="backfill_graph")
                        ents_added += res.get("entities_added", 0)
                        trips_added += res.get("triples_added", 0)
                    new += len(batch_ids)
                    if not args.dry_run:
                        done.update(batch_ids)
                        _save_state(done)
                    batch_texts, batch_sources, batch_ids = [], [], []
            if batch_texts:
                entities, triples = extractor.extract_batch(batch_texts, batch_sources)
                if graph is not None:
                    res = extractor.cognify(graph, entities, triples, source="backfill_graph")
                    ents_added += res.get("entities_added", 0)
                    trips_added += res.get("triples_added", 0)
                new += len(batch_ids)
                if not args.dry_run:
                    done.update(batch_ids)
                    _save_state(done)
            offset += len(ids)
            if len(ids) < PAGE:
                break

    total_entities = len(graph.list_entities(limit=1_000_000)) if graph is not None else -1
    total_rels = len(graph.query_active()) if graph is not None else -1
    print(
        f"[graph-backfill] scanned={scanned} new={new} skipped={skipped} "
        f"entities_added={ents_added} triples_added={trips_added} "
        f"graph_entities={total_entities} graph_relationships={total_rels} "
        f"elapsed={time.time() - t0:.1f}s dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
