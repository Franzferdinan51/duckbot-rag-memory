"""
cli.py — command-line interface for duckbot-rag-memory.

Usage:
    python -m src.cli ingest <paths...>           # ingest markdown files/dirs
    python -m src.cli query <question>             # run a hybrid query
    python -m src.cli stats                        # show collection stats
    python -m src.cli eval <benchmark.jsonl>       # run eval
    python -m src.cli consolidate <days>           # episodic → semantic
    python -m src.cli reset                        # wipe all collections
    python -m src.cli doctor                       # check env + deps

Embedding provider is auto-detected from env. Set DUCKBOT_EMBEDDING to
"openai" | "minimax" | "lmstudio" | "local" to override. The .env file
is read automatically; copy from .env.example.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Allow `python -m src.cli` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from src.embeddings import (
    auto_detect_provider,
    EmbeddingError,
    OpenAIEmbeddings,
    LMStudioEmbeddings,
    LocalEmbeddings,
    MiniMaxEmbeddings,
)
from src.ingest import ingest_paths, write_stats_jsonl, IngestStats
from src.store import MemoryStore
from src.query import hybrid_query, format_results
from src.eval import run_eval, append_history
from src.consolidate import extract_facts_from_chunk, deduplicate_facts


async def _resolve_store_and_embedder() -> tuple[MemoryStore, "EmbeddingProvider"]:
    """Auto-detect embedding provider, then build a store with matching dim.

    Falls back to a 1536-d OpenAI-shaped store if no provider is available
    (so stats and reset still work even when no embeddings can be produced).
    """
    try:
        embedder = await auto_detect_provider()
    except EmbeddingError:
        # No provider available — return a default-shaped store (1536d, OpenAI)
        # so commands like stats and reset still work.
        return MemoryStore(embedding_dim=1536, embedding_provider_name="unconfigured"), None
    # Make sure the dim is resolved (call embed once if provider supports it)
    if embedder.name in ("lmstudio", "local", "minimax"):
        try:
            probe = await embedder.embed_one("dim probe")
            embedder.dim = len(probe)
        except Exception:
            pass
    store = MemoryStore(
        embedding_dim=embedder.dim,
        embedding_provider_name=embedder.name,
    )
    return store, embedder


HISTORY_PATH = Path(__file__).resolve().parent.parent / "data" / "ingest_history.jsonl"
EVAL_HISTORY_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_history.jsonl"


def cmd_ingest(args: argparse.Namespace) -> int:
    async def run() -> IngestStats:
        store, embedder = await _resolve_store_and_embedder()
        if embedder is None:
            raise SystemExit("No embedding provider configured. Set DUCKBOT_EMBEDDING or one of the API keys in .env")
        print(f"  embedding provider: {embedder.name} ({embedder.dim}d)", file=sys.stderr)
        return await ingest_paths(
            args.paths,
            store=store,
            embedder=embedder,
            chunk_size=args.chunk_size,
            overlap_pct=args.overlap,
        )
    stats = asyncio.run(run())
    write_stats_jsonl(stats, HISTORY_PATH)
    print(json.dumps(stats.to_dict(), indent=2))
    return 0 if not stats.errors else 1


def cmd_query(args: argparse.Namespace) -> int:
    async def run():
        store, embedder = await _resolve_store_and_embedder()
        if embedder is None:
            raise SystemExit("No embedding provider configured. Set DUCKBOT_EMBEDDING or one of the API keys in .env")
        results, stats = await hybrid_query(
            args.question, store, embedder,
            n_results=args.n, tier=None,
        )
        return results, stats
    results, stats = asyncio.run(run())
    print(format_results(results, max_chars=args.max_chars))
    sys.stderr.write(json.dumps(stats.to_dict()) + "\n")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    async def run():
        store, _ = await _resolve_store_and_embedder()
        return store.stats().to_dict()
    print(json.dumps(asyncio.run(run()), indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    summary = asyncio.run(run_eval(args.benchmark))
    append_history(summary, EVAL_HISTORY_PATH)
    print(json.dumps(summary.to_dict(), indent=2))
    return 0


def cmd_consolidate(args: argparse.Namespace) -> int:
    """Naive consolidation: pull episodic chunks, extract facts, dedupe, log.
    Doesn't actually add to semantic tier yet (no LLM extraction)."""
    async def run():
        store, _ = await _resolve_store_and_embedder()
        return store
    store = asyncio.run(run())
    coll = store.collection_for(__import__("src.tier", fromlist=["Tier"]).Tier.EPISODIC)
    # Pull last N chunks (sorted by ingested_at desc, where=recent)
    recent = coll.get(
        limit=200,
        include=["documents", "metadatas"],
    )
    if not recent or not recent.get("ids"):
        print("No episodic chunks to consolidate.")
        return 0
    all_facts = []
    for i, chunk_id in enumerate(recent["ids"]):
        facts = extract_facts_from_chunk(
            recent["documents"][i],
            chunk_id,
            recent["metadatas"][i].get("source_path", "<unknown>"),
        )
        all_facts.extend(facts)
    deduped = deduplicate_facts(all_facts)
    print(json.dumps({
        "episodic_chunks_scanned": len(recent["ids"]),
        "facts_extracted": len(all_facts),
        "facts_after_dedup": len(deduped),
        "sample_facts": [f.to_dict() for f in deduped[:10]],
    }, indent=2))
    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    if not args.yes:
        print("Refusing to reset without --yes", file=sys.stderr)
        return 1
    async def run():
        store, _ = await _resolve_store_and_embedder()
        return store
    asyncio.run(run()).reset()
    print("All collections reset.")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Sanity check: env, deps, store."""
    import importlib
    checks = []
    # 1. Python version
    checks.append(("python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}", True))
    # 2. Critical deps
    for mod in ["chromadb", "httpx", "numpy"]:
        try:
            importlib.import_module(mod)
            checks.append((mod, "imported", True))
        except ImportError as exc:
            checks.append((mod, str(exc), False))
    # 3. Optional deps
    for mod in ["sentence_transformers"]:
        try:
            importlib.import_module(mod)
            checks.append((mod, "imported (local mode available)", True))
        except ImportError:
            checks.append((mod, "not installed (local mode disabled)", False))
    # 4. Env vars (all 4 providers)
    checks.append(("OPENAI_API_KEY", "set" if os.environ.get("OPENAI_API_KEY") else "MISSING", bool(os.environ.get("OPENAI_API_KEY"))))
    checks.append(("MINIMAX_API_KEY", "set" if os.environ.get("MINIMAX_API_KEY") else "MISSING", bool(os.environ.get("MINIMAX_API_KEY"))))
    checks.append(("LMSTUDIO_URL", os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234/v1"), True))
    # 5. LM Studio reachability
    lm_url = os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234/v1")
    try:
        import httpx
        with httpx.Client(timeout=2.0) as c:
            r = c.get(f"{lm_url.rstrip('/v1')}/v1/models")
            lm_ok = r.status_code == 200
            lm_info = f"reachable ({r.status_code})" if lm_ok else f"unreachable ({r.status_code})"
    except Exception as exc:
        lm_ok = False
        lm_info = f"unreachable ({exc})"
    checks.append(("LM Studio", lm_info, lm_ok))
    # 6. Store reachable (default dim)
    try:
        async def _check():
            store, emb = await _resolve_store_and_embedder()
            return store, emb
        store, emb = asyncio.run(_check())
        stats = store.stats()
        tiers_with_data = sum(1 for t in ['working','episodic','semantic','procedural'] if getattr(stats, t, 0) > 0)
        checks.append(("chroma store", f"{stats.total} chunks across {tiers_with_data} tiers (provider={emb.name}, dim={emb.dim})", True))
    except Exception as exc:
        checks.append(("chroma store", str(exc), False))

    max_name = max(len(c[0]) for c in checks)
    all_ok = True
    for name, value, ok in checks:
        marker = "✓" if ok else "✗"
        print(f"  {marker} {name.ljust(max_name)}  {value}")
        if not ok:
            all_ok = False
    return 0 if all_ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="duckbot-rag-memory",
        description="RAG + memory system for OpenClaw/Hermes agent memory",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_ingest = sub.add_parser("ingest", help="ingest markdown files/dirs")
    p_ingest.add_argument("paths", nargs="+", help="files or directories to ingest")
    p_ingest.add_argument("--chunk-size", type=int, default=512, help="target tokens per chunk")
    p_ingest.add_argument("--overlap", type=float, default=0.15, help="overlap fraction")
    p_ingest.set_defaults(func=cmd_ingest)

    p_query = sub.add_parser("query", help="run a hybrid query")
    p_query.add_argument("question", help="the query text")
    p_query.add_argument("-n", type=int, default=5, help="number of results")
    p_query.add_argument("--max-chars", type=int, default=400, help="preview chars per result")
    p_query.set_defaults(func=cmd_query)

    p_stats = sub.add_parser("stats", help="show store stats")
    p_stats.set_defaults(func=cmd_stats)

    p_eval = sub.add_parser("eval", help="run retrieval eval")
    p_eval.add_argument("benchmark", help="path to benchmark JSONL")
    p_eval.set_defaults(func=cmd_eval)

    p_consol = sub.add_parser("consolidate", help="episodic → semantic distillation")
    p_consol.add_argument("days", nargs="?", type=int, default=7, help="days of episodic to consolidate")
    p_consol.set_defaults(func=cmd_consolidate)

    p_reset = sub.add_parser("reset", help="wipe all collections (DANGEROUS)")
    p_reset.add_argument("--yes", action="store_true", help="confirm")
    p_reset.set_defaults(func=cmd_reset)

    p_doc = sub.add_parser("doctor", help="check env + deps")
    p_doc.set_defaults(func=cmd_doctor)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())