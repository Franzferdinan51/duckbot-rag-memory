"""
cli.py — command-line interface for duckbot-rag-memory.

Usage:
    python -m src.cli ingest <paths...>           # ingest markdown files/dirs
    python -m src.cli query <question>             # run a hybrid query
    python -m src.cli stats                        # show collection stats
    python -m src.cli eval <benchmark.jsonl>       # run eval
    python -m src.cli consolidate <days>           # episodic → semantic
    python -m src.cli reset                        # wipe all collections
    python -m src.cli compact                      # dedupe + VACUUM the Chroma store
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
    make_query_embedder,
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
    # Make sure the dim is resolved (call embed once if provider supports it).
    # v0.11.2: use the embeddings module's cache so CLI invocations in the
    # same process don't re-probe LM Studio. The first CLI call still
    # resolves the dim; subsequent calls (e.g. in test suites or chained
    # commands) get the cached value with no network call.
    if embedder.name in ("lmstudio", "local", "minimax"):
        from src.embeddings import _EMBEDDER_DIM_CACHE
        cache_key = (getattr(embedder, "base_url", ""), getattr(embedder, "model", ""))
        cached = _EMBEDDER_DIM_CACHE.get(cache_key)
        if cached is not None and cached > 0:
            embedder.dim = cached
        else:
            try:
                probe = await embedder.embed_one("dim probe")
                if probe:
                    embedder.dim = len(probe)
                    _EMBEDDER_DIM_CACHE[cache_key] = embedder.dim
            except Exception:
                pass
    store = MemoryStore(
        embedding_dim=embedder.dim,
        embedding_provider_name=embedder.name,
    )
    return store, embedder


HISTORY_PATH = Path(__file__).resolve().parent.parent / "data" / "ingest_history.jsonl"
EVAL_HISTORY_PATH = Path(__file__).resolve().parent.parent / "data" / "eval_history.jsonl"


def _load_dotenv() -> None:
    """Load .env from the repo root into os.environ. Idempotent. Silent on missing."""
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if not env_file.exists():
        return
    try:
        # Prefer python-dotenv if installed
        from dotenv import load_dotenv
        load_dotenv(env_file, override=False)
        return
    except ImportError:
        pass
    # Fallback: manual parse (handles KEY=VALUE with optional quotes / comments)
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv()


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
        # Use query-optimized embedding pass for MiniMax (different vectors than ingest)
        query_emb = make_query_embedder(embedder)
        results, stats = await hybrid_query(
            args.question, store, query_emb,
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


def cmd_wake_up(args: argparse.Namespace) -> int:
    """One-call session-start context load (MemPalace-inspired).

    Delegates to Brain.wake_up() and prints a formatted markdown block
    ready to paste into an agent's context. Designed for use as a
    Hermes / OpenClaw session-start hook — `scripts/hermes-preflight.sh`
    shells out to this.

    With --json: prints the full wake_up() result as JSON (one line)
    for programmatic consumers (e.g. agent runtimes that pipe the
    output into a parser).
    """
    from src.connectors.base import Brain
    brain = Brain()
    result = brain.wake_up(
        query=getattr(args, "query", None),
        k=getattr(args, "k", 8),
        include_blocks=getattr(args, "include_blocks", True),
        include_graph=getattr(args, "include_graph", True),
        include_fsrs_review=getattr(args, "include_fsrs_review", True),
    )
    if getattr(args, "json", False):
        import json
        print(json.dumps(result, indent=2, default=str))
        return 0
    # Pretty-print as a markdown block so agents can paste it into context.
    lines = ["# 🧠 Brain Wake-Up", ""]
    memories = result.get("memories") or []
    if memories:
        lines.append(f"## Recent Memories ({len(memories)})")
        lines.append("")
        for m in memories[:8]:
            text = (m.get("text") or "")[:280].replace("\n", " ")
            tier = m.get("tier", "?")
            lines.append(f"- **[{tier}]** {text}")
        lines.append("")
    blocks = result.get("blocks") or []
    if blocks:
        lines.append(f"## Active Memory Blocks ({len(blocks)})")
        lines.append("")
        for b in blocks:
            lines.append(f"- **{b.get('name','')}** ({b.get('char_count',0)} chars): {b.get('preview','')}")
        lines.append("")
    graph = result.get("graph_summary") or {}
    if graph.get("top_entities"):
        lines.append(f"## Graph ({graph.get('entity_count',0)} entities)")
        for e in graph["top_entities"][:5]:
            lines.append(f"- {e.get('name','?')} ({e.get('kind','?')})")
        lines.append("")
    queue = result.get("fsrs_review_queue") or []
    if queue:
        lines.append(f"## FSRS Review Queue ({len(queue)} due)")
        for q in queue[:5]:
            lines.append(f"- {q.get('chunk_id','?')} R={q.get('retrievability',0):.2f}")
        lines.append("")
    print("\n".join(lines))
    return 0


def cmd_reflect(args: argparse.Namespace) -> int:
    """Sleep-time consolidation. Wrapper around Memory.reflect()."""
    async def run():
        from src.memory import Memory
        return await Memory().reflect(lookback_days=args.days, max_chunks=args.max_chunks)
    result = asyncio.run(run())
    print(json.dumps(result, indent=2))
    return 0


def cmd_palace(args: argparse.Namespace) -> int:
    """Wing/Room/Drawer 2D view of the brain (MemPalace-inspired).

    With no --wing: list every wing. With --wing: walk that wing.
    Cross-references wings against the 'user' memory block.
    """
    async def run():
        from src.mcp_server import handle_brain_palace
        return await handle_brain_palace({
            "wing": args.wing,
            "room": getattr(args, "room", None),
            "tier": getattr(args, "tier", None),
            "max_drawers": args.max_drawers,
        })
    result = asyncio.run(run())
    print(json.dumps(result, indent=2))
    return 0


def cmd_nudge(args: argparse.Namespace) -> int:
    """Proactive memory nudge: surface stale-but-important memories."""
    async def run():
        from src.mcp_server import handle_brain_nudge
        return await handle_brain_nudge({
            "context": getattr(args, "context", None),
            "k": args.k,
            "min_importance": args.min_importance,
            "stale_days": args.stale_days,
        })
    result = asyncio.run(run())
    print(json.dumps(result, indent=2))
    return 0


def cmd_optimize_fsrs(args: argparse.Namespace) -> int:
    """Self-tune the FSRS-6 forgetting-curve exponent (w20).

    With --apply: also call brain_fsrs_optimize_apply which only
    commits if the fit improves the MSE by at least --min-improvement-pct.
    Use on a weekly cron.
    """
    if args.apply:
        async def run_apply():
            from src.mcp_server import handle_brain_fsrs_optimize_apply
            return await handle_brain_fsrs_optimize_apply({
                "min_improvement_pct": args.min_improvement_pct,
                "w20_lo": args.w20_lo,
                "w20_hi": args.w20_hi,
                "w20_step": args.w20_step,
            })
        result = asyncio.run(run_apply())
    else:
        async def run():
            from src.mcp_server import handle_brain_optimize_fsrs
            return await handle_brain_optimize_fsrs({
                "default_w20": args.default_w20,
                "w20_lo": args.w20_lo,
                "w20_hi": args.w20_hi,
                "w20_step": args.w20_step,
            })
        result = asyncio.run(run())
    print(json.dumps(result, indent=2))
    return 0


def cmd_apply_fsrs_w20(args: argparse.Namespace) -> int:
    """Apply a chosen w20 to the brain. Persists via DUCKBOT_FSRS_W20 env var."""
    async def run():
        from src.mcp_server import handle_brain_apply_fsrs_w20
        return await handle_brain_apply_fsrs_w20({"w20": args.w20})
    result = asyncio.run(run())
    print(json.dumps(result, indent=2))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export the entire brain as a single markdown file."""
    async def run():
        from src.mcp_server import handle_brain_export
        return await handle_brain_export({
            "out_path": args.out_path,
            "tier": args.tier,
            "include_superseded": args.include_superseded,
        })
    result = asyncio.run(run())
    print(json.dumps(result, indent=2))
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    """Import a markdown file into the brain."""
    async def run():
        from src.mcp_server import handle_brain_import
        return await handle_brain_import({
            "in_path": args.in_path,
            "source_path": args.source_path,
        })
    result = asyncio.run(run())
    print(json.dumps(result, indent=2))
    return 0


def cmd_seed_demo(args: argparse.Namespace) -> int:
    """Seed the brain with a small bundled sample corpus."""
    async def run():
        from src.mcp_server import handle_brain_seed_demo
        return await handle_brain_seed_demo({"force": args.force})
    result = asyncio.run(run())
    print(json.dumps(result, indent=2))
    return 0


def cmd_decay(args: argparse.Namespace) -> int:
    """Memory decay: prune chunks below the Ebbinghaus retention floor.

    Default is dry-run (preview only). Pass --apply to actually delete.
    Daily cron: `python -m src.cli decay --apply --retention-floor 0.05`.
    """
    async def run():
        from src.mcp_server import handle_brain_decay_apply
        return await handle_brain_decay_apply({
            "tier": args.tier,
            "retention_floor": args.retention_floor,
            "max_prune": args.max_prune,
            "dry_run": not args.apply,
        })
    result = asyncio.run(run())
    print(json.dumps(result, indent=2))
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Consolidated entity view: graph + recent memories + blocks.
    Returns everything the brain knows about one entity in one dict."""
    from src.connectors.base import Brain
    brain = Brain()
    result = brain.inspect(entity=args.entity, k=args.k)
    print(json.dumps(result, indent=2, default=str))
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


def cmd_compact(args: argparse.Namespace) -> int:
    """Dedupe + vacuum the Chroma store.

    Chroma's SQLite WAL mode grows unboundedly, and `add_chunks()` with
    `upsert` semantics can leave duplicate ids in edge cases (e.g. when
    a chunk's `id` field collides on re-ingest after schema change).
    This command:
      1. Scans every tier collection for duplicate ids.
      2. Deduplicates by keeping the most recently-ingested copy.
      3. Reports disk usage before + after.
      4. (Optional) Runs `VACUUM` on the underlying SQLite db.

    Cross-platform: works on macOS / Linux / Windows. The Chroma
    PersistentClient handles path translation; on Windows we just
    need a Path (which pathlib does correctly).
    """
    async def run():
        from src.store import MemoryStore
        return MemoryStore()

    store = asyncio.run(run())
    backend = store.backend
    if not hasattr(backend, "_client"):
        print("❌ compact only works with the Chroma backend.", file=sys.stderr)
        print(f"   Current backend: {backend.name}", file=sys.stderr)
        return 1

    client = backend._client
    persist_dir = backend.persist_dir
    print(f"Compacting Chroma store at {persist_dir}...")

    # 1. Get disk usage before
    def dir_size(p: Path) -> int:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

    before_size = dir_size(persist_dir) if persist_dir.exists() else 0
    total_dups = 0
    total_kept = 0
    for tier_name in backend.supported_tiers:
        coll = backend.collection_for(tier_name)
        try:
            resp = coll.get(include=["metadatas"])
        except Exception as e:
            print(f"  [{tier_name}] skip: {e}")
            continue
        ids = (resp or {}).get("ids") or []
        metas = (resp or {}).get("metadatas") or []
        if not ids:
            continue
        # Group by id, keep the one with the highest ingested_at
        by_id: dict[str, tuple[int, int, dict]] = {}
        for i, cid in enumerate(ids):
            m = metas[i] if i < len(metas) else {}
            ts = int(m.get("ingested_at") or 0)
            cur = by_id.get(cid)
            if cur is None or ts > cur[0]:
                by_id[cid] = (ts, i, m)
        # Dedupe: any id that appears more than once has duplicates
        from collections import Counter
        id_counts = Counter(ids)
        dup_ids = [cid for cid, c in id_counts.items() if c > 1]
        if dup_ids:
            # Re-upsert the kept version, which will replace the dupes
            keep_ids = list(by_id.keys())
            print(f"  [{tier_name}] {len(ids)} chunks, {len(dup_ids)} duplicate ids, keeping {len(keep_ids)}")
            # The easiest dedupe: re-upsert the latest copy of each id
            keep_resp = coll.get(ids=keep_ids, include=["documents", "embeddings", "metadatas"])
            kr_ids = keep_resp["ids"]
            kr_docs = keep_resp["documents"]
            kr_embs = keep_resp["embeddings"]
            kr_metas = keep_resp["metadatas"]
            coll.upsert(ids=kr_ids, documents=kr_docs, embeddings=kr_embs, metadatas=kr_metas)
            total_dups += len(dup_ids)
            total_kept += len(keep_ids)
        else:
            total_kept += len(ids)
            print(f"  [{tier_name}] {len(ids)} chunks, no duplicates")

    # 2. Try to vacuum the SQLite db (cross-platform; Windows uses the
    #    same sqlite3 module).
    try:
        # Find the sqlite file. Chroma stores it at <persist_dir>/chroma.sqlite3
        sqlite_path = persist_dir / "chroma.sqlite3"
        if sqlite_path.exists():
            import sqlite3
            # VACUUM cannot run inside an active transaction; sqlite3's
            # context manager opens an implicit txn. Use autocommit mode.
            conn = sqlite3.connect(str(sqlite_path), isolation_level=None)
            try:
                conn.execute("VACUUM")
            finally:
                conn.close()
            print(f"  VACUUM'd {sqlite_path}")
    except Exception as e:
        print(f"  VACUUM skipped: {e}", file=sys.stderr)

    # 3. Report after
    after_size = dir_size(persist_dir) if persist_dir.exists() else 0
    saved_mb = (before_size - after_size) / 1024 / 1024
    print()
    print(f"✓ Compact complete: {total_dups} duplicates removed, {total_kept} chunks kept")
    print(f"  Disk: {before_size / 1024 / 1024:.1f} MB → {after_size / 1024 / 1024:.1f} MB (saved {saved_mb:.1f} MB)")
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
    lm_key = (
        os.environ.get("LMSTUDIO_API_KEY")
        or os.environ.get("LMSTUDIO_KEY")
        or os.environ.get("LM_API_TOKEN")
        or ""
    )
    try:
        import httpx
        headers = {"Authorization": f"Bearer {lm_key}"} if lm_key else {}
        with httpx.Client(timeout=2.0) as c:
            r = c.get(f"{lm_url.rstrip('/v1')}/v1/models", headers=headers)
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




def cmd_dashboard(args: argparse.Namespace) -> int:
    """Print a human-readable dashboard of the brain's current state."""
    from .dashboard import build_report, format_report
    watcher_log = Path(__file__).resolve().parent.parent / "data" / "watcher.log"
    r = build_report(watcher_log=watcher_log)
    if getattr(args, "json", False):
        import json
        print(json.dumps(r.to_dict(), indent=2, default=str))
    else:
        print(format_report(r))
    return 0




def cmd_hermes(args: argparse.Namespace) -> int:
    """Hermes CLI shim: 'python -m src.cli hermes <verb> [args...]' delegates to the connectors.hermes module."""
    from .connectors import hermes
    return hermes.main(args.verb + [args.remainder] if hasattr(args, "remainder") and args.remainder else args.verb)


async def _run_brain_sync(target: str, memory_k: int, user_k: int) -> dict:
    """Run brain_sync without needing the MCP stdio transport."""
    from .mcp_server import handle_brain_sync
    return await handle_brain_sync({
        "target": target,
        "memory_k": memory_k,
        "user_k": user_k,
        "dry_run": False,
    })


def cmd_sync(args: argparse.Namespace) -> int:
    """Sync stored memories to OpenClaw and/or Hermes agent context files.

    This is what makes the enhanced brain work: it writes memories back to
    the files that agents read at startup (MEMORY.md, USER.md, SOUL.md).
    Call this after ingest or on a cron to keep context files fresh.
    """
    import asyncio
    from .mcp_server import handle_brain_sync
    result = asyncio.run(handle_brain_sync({
        "target": args.target,
        "memory_k": args.memory_k,
        "user_k": args.user_k,
        "dry_run": args.dry_run,
    }))
    import json
    print(json.dumps(result, indent=2, default=str))
    return 0


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
    p_compact = sub.add_parser("compact", help="dedupe + VACUUM the Chroma store (cross-platform)")
    p_compact.set_defaults(func=cmd_compact)

    p_doc = sub.add_parser("doctor", help="check env + deps")
    p_doc.set_defaults(func=cmd_doctor)

    # Wake-up: session-start context load (Hermes pre-flight / OpenClaw init).
    p_wakeup = sub.add_parser(
        "wake-up",
        help="one-call session-start context load (memories + blocks + graph + FSRS queue)",
    )
    p_wakeup.add_argument("--query", help="optional anchor query for recall")
    p_wakeup.add_argument("-k", type=int, default=8, help="max memories")
    p_wakeup.add_argument("--no-blocks", action="store_false", dest="include_blocks")
    p_wakeup.add_argument("--no-graph", action="store_false", dest="include_graph")
    p_wakeup.add_argument("--no-fsrs-review", action="store_false", dest="include_fsrs_review")
    p_wakeup.add_argument("--json", action="store_true",
                          help="output as JSON (default: markdown block)")
    p_wakeup.set_defaults(
        func=cmd_wake_up,
        include_blocks=True,
        include_graph=True,
        include_fsrs_review=True,
    )

    # Reflect: sleep-time consolidation (Hermes post-flight).
    p_reflect = sub.add_parser(
        "reflect",
        help="sleep-time consolidation (episodic → semantic distillation)",
    )
    p_reflect.add_argument("--days", type=int, default=7, help="lookback days")
    p_reflect.add_argument("--max-chunks", type=int, default=200, help="max episodic chunks to scan")
    p_reflect.set_defaults(func=cmd_reflect)

    # Palace: Wing/Room/Drawer 2D view (MemPalace-inspired).
    p_palace = sub.add_parser(
        "palace",
        help="wing/room/drawer 2D view of the brain",
    )
    p_palace.add_argument("--wing", help="walk a specific wing (person/project)")
    p_palace.add_argument("--room", help="filter to one room (date or filename)")
    p_palace.add_argument("--tier", choices=["working", "episodic", "semantic", "procedural"],
                          help="filter to one tier")
    p_palace.add_argument("--max-drawers", type=int, default=100, help="cap on drawers returned")
    p_palace.set_defaults(func=cmd_palace)

    # Nudge: proactive memory nudge.
    p_nudge = sub.add_parser(
        "nudge",
        help="proactive memory nudge (stale-but-important)",
    )
    p_nudge.add_argument("--context", help="optional current focus — biases toward relevant memories")
    p_nudge.add_argument("-k", type=int, default=5, help="max memories")
    p_nudge.add_argument("--min-importance", type=float, default=0.6, help="importance threshold (0..1)")
    p_nudge.add_argument("--stale-days", type=int, default=7, help="consider stale if last_recalled_at older than this many days")
    p_nudge.set_defaults(func=cmd_nudge)

    # Optimize FSRS: self-tune w20.
    p_opt = sub.add_parser(
        "optimize-fsrs",
        help="self-tune the FSRS-6 forgetting-curve exponent (with --apply, commits if better)",
    )
    p_opt.add_argument("--default-w20", type=float, default=0.9, help="comparison baseline")
    p_opt.add_argument("--w20-lo", type=float, default=0.05, help="search grid low")
    p_opt.add_argument("--w20-hi", type=float, default=3.0, help="search grid high")
    p_opt.add_argument("--w20-step", type=float, default=0.05, help="search grid step")
    p_opt.add_argument("--apply", action="store_true",
                      help="commit the new w20 if it improves the baseline by at least --min-improvement-pct")
    p_opt.add_argument("--min-improvement-pct", type=float, default=1.0,
                      help="minimum improvement over baseline (percent) to apply")
    p_opt.set_defaults(func=cmd_optimize_fsrs)

    # Memory decay: prune chunks below retention floor.
    p_decay = sub.add_parser(
        "decay",
        help="memory decay: prune chunks below the Ebbinghaus retention floor (with --apply, deletes)",
    )
    p_decay.add_argument("--tier", choices=["working", "episodic", "semantic", "procedural"],
                         help="limit to one tier (default: all)")
    p_decay.add_argument("--retention-floor", type=float, default=0.05,
                         help="chunks with R < floor are pruned (default 0.05)")
    p_decay.add_argument("--max-prune", type=int, default=1000,
                         help="safety cap on chunks deleted in one call")
    p_decay.add_argument("--apply", action="store_true",
                         help="actually delete (default: dry-run — preview only)")
    p_decay.set_defaults(func=cmd_decay)

    # Inspect: consolidated entity view.
    p_inspect = sub.add_parser(
        "inspect",
        help="consolidated entity view: graph + recent memories + blocks",
    )
    p_inspect.add_argument("entity", help="entity name to inspect (e.g. 'Duckets', 'OpenClaw')")
    p_inspect.add_argument("-k", type=int, default=10, help="max memories to recall")
    p_inspect.set_defaults(func=cmd_inspect)

    # Apply FSRS w20: persist the new value.
    p_apply = sub.add_parser(
        "apply-fsrs-w20",
        help="apply a chosen w20 (persists via DUCKBOT_FSRS_W20 env var)",
    )
    p_apply.add_argument("w20", type=float, help="the new w20 to use")
    p_apply.set_defaults(func=cmd_apply_fsrs_w20)

    # Export the brain as a single markdown file.
    p_export = sub.add_parser(
        "export",
        help="export the brain as a single markdown file (data/brain_export.md)",
    )
    p_export.add_argument("--out-path", default="data/brain_export.md",
                         help="where to write the export (default: data/brain_export.md)")
    p_export.add_argument("--tier", choices=["working", "episodic", "semantic", "procedural"],
                         help="export only one tier (default: all)")
    p_export.add_argument("--include-superseded", action="store_true",
                         help="include chunks marked superseded_by (default: skip)")
    p_export.set_defaults(func=cmd_export)

    # Import a markdown file into the brain.
    p_import = sub.add_parser(
        "import",
        help="import a markdown file (## sections → chunks) into the brain",
    )
    p_import.add_argument("in_path", help="path to the markdown file to import")
    p_import.add_argument("--source-path", help="stamped as source_path on every chunk (default: filename)")
    p_import.set_defaults(func=cmd_import)

    # Seed the brain with bundled demo data.
    p_seed = sub.add_parser(
        "seed-demo",
        help="seed the brain with a small bundled sample corpus",
    )
    p_seed.add_argument("--force", action="store_true",
                        help="re-seed even if chunks already exist")
    p_seed.set_defaults(func=cmd_seed_demo)

    p_hermes = sub.add_parser("hermes", help="Hermes agent CLI shim: hermes <verb> [args...]")
    p_hermes.add_argument("verb", nargs="+", help="verb (remember, recall, stats, etc.) + args")
    p_hermes.set_defaults(func=cmd_hermes)

    p_dash = sub.add_parser("dashboard", help="brain observability dashboard")
    p_dash.add_argument("--json", action="store_true", help="output as JSON")
    p_dash.set_defaults(func=cmd_dashboard)

    p_sync = sub.add_parser("sync", help="Sync stored memories to OpenClaw/Hermes context files (enhanced brain)")
    p_sync.add_argument("--target", choices=["openclaw", "hermes", "both"], default="both",
                        help="which agent to sync (default: both)")
    p_sync.add_argument("--memory-k", type=int, default=20, help="max memories per tier for MEMORY.md")
    p_sync.add_argument("--user-k", type=int, default=15, help="max facts for USER.md")
    p_sync.add_argument("--dry-run", action="store_true", help="preview without writing files")
    p_sync.set_defaults(func=cmd_sync)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())