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
import shutil
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
    # Validate benchmark file exists before launching the async eval
    # loop — otherwise a missing file surfaces as a raw Python traceback
    # instead of a clean error.
    from pathlib import Path as _Path
    bench_path = _Path(args.benchmark)
    if not bench_path.exists():
        print(json.dumps({"error": f"benchmark file not found: {bench_path}"}))
        return 2
    try:
        summary = asyncio.run(run_eval(args.benchmark))
    except FileNotFoundError as exc:
        print(json.dumps({"error": str(exc)}))
        return 2
    append_history(summary, EVAL_HISTORY_PATH)
    print(json.dumps(summary.to_dict(), indent=2))
    # Compute and print a trend if we have enough history. Surface in
    # JSON so callers can diff against prior runs without parsing stdout.
    try:
        from src.eval import load_history, compute_trend
        history = load_history(EVAL_HISTORY_PATH)
        trend = compute_trend(history)
        if trend["n_runs"] >= 2:
            print("\n--- trend ---")
            print(json.dumps({"trend": trend}, indent=2))
    except Exception as exc:
        # Non-fatal — trend is a nice-to-have, don't break the eval.
        print(f"\n(trend unavailable: {exc})", file=sys.stderr)
    return 0


def cmd_consolidate(args: argparse.Namespace) -> int:
    """Sleep-time consolidation. Wrapper around Memory.reflect()."""
    async def run():
        from src.memory import Memory
        return await Memory().reflect(lookback_days=args.days, max_chunks=200)
    result = asyncio.run(run())
    print(json.dumps(result, indent=2))
    return 0


def cmd_wake_up(args: argparse.Namespace) -> int:
    """One-call session-start context load (MemPalace-inspired).

    Delegates to Brain.wake_up() and prints a formatted markdown block
    ready to paste into an agent's context. The MemoryProvider plugin's
    on_session_start hook calls this automatically; use this CLI verb
    for manual / cron invocations via `scripts/hermes-preflight.sh`.

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
    store = asyncio.run(run())
    store.reset()
    # Also wipe the on-disk data directory.  ChromaDB's reset() only
    # unregisters collections from its registry but leaves segment files
    # on disk; those files may carry a stale schema that causes
    # "metadata segment reader: column 0 mismatched types" errors.
    # Wiping the directory guarantees a clean slate.
    try:
        persist_dir = store._backend.persist_dir
        if persist_dir.exists():
            shutil.rmtree(persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)
            print(f"Wiped and recreated {persist_dir}.")
    except Exception as e:
        print(f"Warning: could not wipe persist dir: {e}", file=sys.stderr)
    print("All collections reset.")
    return 0


# ---------------------------------------------------------------------------
# Maintenance commands (v0.15.2) — fsck / vacuum / reindex / prune / etc.
# ---------------------------------------------------------------------------

def cmd_fsck(args: argparse.Namespace) -> int:
    """Per-collection health report.

    Reports on-disk size, vector count, bytes/vector, and flags any
    collection that looks like it has HNSW bloat (>100 KB/vector) or
    that predates the explicit-HNSW-params fix.
    """
    async def run():
        store, _ = await _resolve_store_and_embedder()
        return store._backend.fsck()
    report = asyncio.run(run())
    print(json.dumps(report, indent=2, default=str))
    # Exit non-zero if there are any issues, so cron/operators can alert.
    return 1 if report.get("issues") else 0


def cmd_vacuum(args: argparse.Namespace) -> int:
    """Drop a single tier's ChromaDB collection (frees disk immediately).

    The collection is recreated on the next add_chunks() call with the
    current (fixed) HNSW params. Use `reindex-tier` to rebuild content
    from the watcher state.
    """
    tier = args.tier
    if not args.yes:
        print(f"Refusing to vacuum tier={tier!r} without --yes", file=sys.stderr)
        return 1
    async def run():
        store, _ = await _resolve_store_and_embedder()
        return store._backend.vacuum_tier(tier)
    result = asyncio.run(run())
    print(json.dumps(result, indent=2, default=str))
    print(
        f"\nVacuumed tier={tier!r}. Run `python -m src.cli reindex-tier {tier}` "
        f"to re-ingest from watcher state, or `python -m src.cli wake-up` to "
        f"rehydrate from the live system.",
        file=sys.stderr,
    )
    return 0


def cmd_reindex_tier(args: argparse.Namespace) -> int:
    """Wipe a tier and re-ingest from the watcher state (or live sources).

    Re-ingests every source_path tracked in `data/watcher_state.json`
    for files that originally landed in the target tier. Tier routing
    uses the existing tier.py classifier. Chunks that no longer match
    the target tier are skipped (so reindexing semantic doesn't dump
    procedural chunks back into semantic).
    """
    tier = args.tier
    if not args.yes:
        print(f"Refusing to reindex-tier {tier!r} without --yes", file=sys.stderr)
        return 1
    # 1. Vacuum the tier.
    async def run():
        store, _ = await _resolve_store_and_embedder()
        return store._backend.vacuum_tier(tier)
    asyncio.run(run())
    print(f"Vacuumed tier={tier!r}. Now re-ingesting from watcher state…",
          file=sys.stderr)

    # 2. Load watcher state and re-ingest each file.
    state_path = Path("data/watcher_state.json")
    if not state_path.exists():
        print("No watcher state — nothing to reindex. Use `ingest <paths>` "
              "to import fresh sources.", file=sys.stderr)
        return 0
    try:
        state = json.loads(state_path.read_text())
    except Exception as e:
        print(f"Failed to read watcher state: {e}", file=sys.stderr)
        return 1
    files = list((state.get("files") or {}).keys())
    if not files:
        print("Watcher state has no files.", file=sys.stderr)
        return 0

    # 3. Run the ingest pipeline.
    cmd_args = argparse.Namespace(
        paths=files,
        chunk_size=getattr(args, "chunk_size", 512),
        overlap=getattr(args, "overlap", 0.15),
    )
    return cmd_ingest(cmd_args)


def cmd_prune_empty_collections(args: argparse.Namespace) -> int:
    """Delete ChromaDB collections that are empty AND not in the
    declared tier list. Cleans up orphaned collections left behind by
    past schema renames."""
    if not args.yes:
        print("Refusing to prune without --yes", file=sys.stderr)
        return 1
    async def run():
        store, _ = await _resolve_store_and_embedder()
        return store._backend.prune_empty_collections()
    result = asyncio.run(run())
    print(json.dumps(result, indent=2, default=str))
    if not result.get("deleted"):
        print("No empty non-tier collections to prune.", file=sys.stderr)
    return 0


def cmd_purge_quarantine(args: argparse.Namespace) -> int:
    """Delete quarantined items older than `--older-than-days` (default 30).

    Items in `data/quarantine.db` are stored with an `added_at` timestamp.
    Old quarantines pile up; this lets operators age them out without
    touching the live store.
    """
    from datetime import datetime, timezone
    qpath = Path(os.environ.get(
        "DUCKBOT_QUARANTINE_PATH", "data/quarantine.db",
    ))
    if not qpath.exists():
        print("No quarantine DB found.", file=sys.stderr)
        return 0
    try:
        import sqlite3
        conn = sqlite3.connect(str(qpath))
        cutoff = datetime.now(timezone.utc).timestamp() - (args.older_than_days * 86400)
        cur = conn.execute("DELETE FROM quarantine WHERE added_at < ?", (cutoff,))
        deleted = cur.rowcount
        conn.commit()
        # VACUUM the file to reclaim space (SQLite stores deleted rows
        # in the file until VACUUM runs).
        conn.execute("VACUUM")
        conn.close()
        print(f"Deleted {deleted} quarantined items older than {args.older_than_days} days.")
        print(f"VACUUMed {qpath}.")
    except Exception as e:
        print(f"Purge failed: {e}", file=sys.stderr)
        return 1
    return 0


def cmd_rotate_events(args: argparse.Namespace) -> int:
    """Rotate `data/events.db` when it exceeds `--max-mb` (default 50).

    Renames the current DB to `events.<ts>.db` and starts fresh. The
    rotated copy is NOT deleted (operators can inspect it). Use
    `DUCKBOT_EVENTS_KEEP_ROTATED=N` to cap retention.
    """
    from datetime import datetime, timezone
    import gzip
    import shutil as _sh
    epath = Path("data/events.db")
    if not epath.exists():
        print("No events.db to rotate.", file=sys.stderr)
        return 0
    size_mb = epath.stat().st_size / (1024 * 1024)
    if size_mb < args.max_mb:
        print(f"events.db is {size_mb:.1f} MB (under {args.max_mb} MB cap). "
              f"No rotation needed.", file=sys.stderr)
        return 0
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = epath.parent / f"events.{ts}.db"
    if args.gzip:
        archive = epath.parent / f"events.{ts}.db.gz"
        with open(epath, "rb") as f_in, gzip.open(archive, "wb") as f_out:
            _sh.copyfileobj(f_in, f_out)
    else:
        epath.rename(archive)
    # Drop the old WAL/SHM so the new file starts clean.
    for ext in ("-wal", "-shm"):
        p = Path(str(epath) + ext)
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass
    # Re-initialize the new DB by touching it (the next record_event()
    # call will lazy-create the schema).
    epath.touch()
    print(f"Rotated {epath} ({size_mb:.1f} MB) → {archive}")
    # Cap retention if DUCKBOT_EVENTS_KEEP_ROTATED is set.
    keep = os.environ.get("DUCKBOT_EVENTS_KEEP_ROTATED")
    if keep and keep.isdigit():
        keep_n = int(keep)
        pattern = "events.*.db*"
        archives = sorted(
            epath.parent.glob(pattern),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in archives[keep_n:]:
            try:
                old.unlink()
                print(f"  pruned old archive: {old.name}")
            except Exception as e:
                print(f"  could not prune {old}: {e}", file=sys.stderr)
    return 0


def cmd_maintenance(args: argparse.Namespace) -> int:
    """Run a safe battery of cleanups in one shot.

    Steps (each idempotent, all safe to run on a live system):
      1. fsck — report health
      2. prune-empty-collections — drop orphaned empty collections
      3. purge-quarantine --older-than-days=30 — age out old quarantines
      4. rotate-events --max-mb=50 — bound the events log
      5. fsck again — show after-state

    Does NOT touch: vacuum, reindex-tier, reset (those are destructive
    and require explicit `--yes`).
    """
    print("=== maintenance: pass 1 (cleanup) ===")
    args.yes = True
    cmd_prune_empty_collections(args)
    args.older_than_days = 30
    cmd_purge_quarantine(args)
    args.max_mb = 50
    args.gzip = True
    cmd_rotate_events(args)
    print()
    print("=== maintenance: pass 2 (fsck after) ===")
    args_fsck = argparse.Namespace()
    return cmd_fsck(args_fsck)


def cmd_update(args: argparse.Namespace) -> int:
    """Check for updates from origin/main and pull if behind.

    Also upgrades deps and runs doctor. Safe to re-run — stashes any
    local changes before pulling and restores them after.

    Returns a dict with keys:
        current_branch, commits_behind, was_updated, had_local_changes,
        doctor_passed, error (if any).
    """
    import subprocess

    def run(*cmd: str, capture: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            cwd=Path(__file__).resolve().parent.parent,
            capture_output=capture,
            text=True,
        )

    result: dict = {}

    # 1. Check we're in a git repo
    git_check = run("git", "rev-parse", "--is-inside-work-tree")
    if git_check.returncode != 0 or "true" not in git_check.stdout.lower():
        print("Not in a git repository.", file=sys.stderr)
        result["error"] = "not a git repo"
        print(json.dumps(result, indent=2))
        return 1

    branch = run("git", "branch", "--show-current").stdout.strip()
    result["current_branch"] = branch

    # 2. Check remote
    remote_check = run("git", "remote", "get-url", "origin")
    if remote_check.returncode != 0:
        print("No remote configured — skipping pull.", file=sys.stderr)
        result["error"] = "no remote"
        print(json.dumps(result, indent=2))
        return 0
    result["remote"] = remote_check.stdout.strip()

    # 3. Check for uncommitted changes
    diff_check = run("git", "diff", "--quiet")
    staged_check = run("git", "diff", "--cached", "--quiet")
    had_changes = diff_check.returncode != 0 or staged_check.returncode != 0
    result["had_local_changes"] = had_changes

    # 4. Fetch + compare
    run("git", "fetch", "origin")
    behind_raw = run("git", "rev-list", "--count", f"HEAD..origin/{branch}")
    try:
        commits_behind = int(behind_raw.stdout.strip())
    except (ValueError, AttributeError):
        commits_behind = 0
    result["commits_behind"] = commits_behind

    if commits_behind == 0:
        result["was_updated"] = False
        result["message"] = "Already up to date."
        print(json.dumps(result, indent=2))
        return 0

    # Dry-run: just report what's available
    if args.dry_run:
        result["was_updated"] = False
        result["message"] = f"{commits_behind} commit(s) behind origin/{branch}. Run without --dry-run to pull."
        print(json.dumps(result, indent=2))
        return 0

    # 5. Stash local changes (if any)
    if had_changes:
        stash_out = run("git", "stash", "push", "-m", "auto-stash before update")
        if stash_out.returncode != 0 and "No local changes" not in stash_out.stderr:
            result["stash_error"] = stash_out.stderr.strip()
            print(f"Warning: stash failed: {stash_out.stderr.strip()}", file=sys.stderr)

    # 6. Pull
    pull = run("git", "pull", "--rebase", "origin", branch)
    if pull.returncode != 0:
        result["was_updated"] = False
        result["pull_error"] = pull.stderr.strip()
        print(f"Pull failed:\n{pull.stderr}", file=sys.stderr)
        if had_changes:
            run("git", "stash", "pop")
        print(json.dumps(result, indent=2))
        return 1

    result["was_updated"] = True
    result["new_head"] = run("git", "log", "-1", "--oneline").stdout.strip()

    # 7. Upgrade deps
    venv_python = _venv_python()
    if venv_python:
        upgrade = run(venv_python, "-m", "pip", "install", "--quiet", "--upgrade", "pip")
        if args.no_deps:
            print("Skipping dep upgrade (--no-deps).")
        else:
            req = Path(__file__).resolve().parent.parent / "requirements.txt"
            if req.exists():
                run(venv_python, "-m", "pip", "install", "-q", "-r", str(req))
                result["deps_upgraded"] = True
            else:
                result["deps_upgraded"] = False

    # 8. Doctor
    if args.no_doctor:
        print("Skipping doctor check (--no-doctor).")
    else:
        checks, all_ok = asyncio.run(build_doctor_checks_async())
        result["doctor_passed"] = all_ok
        result["doctor_checks"] = {name: ok for name, ok, _ in checks}

    # 9. Restore stash
    if had_changes:
        run("git", "stash", "pop")

    print(json.dumps(result, indent=2, default=str))
    return 0


def _venv_python() -> str | None:
    """Return the path to the venv python, or None if not found."""
    repo = Path(__file__).resolve().parent.parent
    for candidate in [
        repo / ".venv" / "bin" / "python",
        repo / ".venv" / "bin" / "python3",
        repo / ".venv" / "Scripts" / "python.exe",
    ]:
        if candidate.exists():
            return str(candidate)
    return None


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
    from src.store import MemoryStore
    store = MemoryStore()
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
            # Batch the upsert to avoid segfaulting ChromaDB's hnswlib/sqlite3
            # native bindings on macOS with large tier re-upserts (same
            # threshold as add_chunks). Override with DUCKBOT_CHROMA_UPSERT_BATCH.
            try:
                batch_size = int(os.environ.get("DUCKBOT_CHROMA_UPSERT_BATCH", 32))
            except (TypeError, ValueError):
                batch_size = 32
            batch_size = max(1, min(batch_size, len(keep_ids)))
            for start in range(0, len(kr_ids), batch_size):
                end = min(start + batch_size, len(kr_ids))
                coll.upsert(
                    ids=kr_ids[start:end],
                    documents=kr_docs[start:end],
                    embeddings=kr_embs[start:end],
                    metadatas=kr_metas[start:end],
                )
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


async def build_doctor_checks_async() -> tuple[list[tuple[str, str, bool]], bool]:
    """Build the shared doctor checklist for CLI and MCP surfaces."""
    import importlib
    checks: list[tuple[str, str, bool]] = []
    required_names = {"chromadb", "httpx", "numpy", "embedding provider", "chroma store"}

    def add_check(name: str, value: str, ok: bool) -> None:
        checks.append((name, value, ok))

    # 1. Python version
    add_check("python", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}", True)
    # 2. Critical deps
    for mod in ["chromadb", "httpx", "numpy"]:
        try:
            importlib.import_module(mod)
            add_check(mod, "imported", True)
        except ImportError as exc:
            add_check(mod, str(exc), False)
    # 3. Optional deps
    try:
        importlib.import_module("sentence_transformers")
        add_check("sentence_transformers", "imported (local mode available)", True)
    except ImportError:
        add_check("sentence_transformers", "not installed (local mode disabled)", True)

    explicit_provider = os.environ.get("DUCKBOT_EMBEDDING", "").lower().strip()
    openai_key = bool(os.environ.get("OPENAI_API_KEY"))
    minimax_key = bool(os.environ.get("MINIMAX_API_KEY"))
    lm_url = os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234/v1")
    add_check("OPENAI_API_KEY", "set" if openai_key else "missing", True)
    add_check("MINIMAX_API_KEY", "set" if minimax_key else "missing", True)
    add_check("LMSTUDIO_URL", lm_url, True)

    # 4. LM Studio reachability
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
            headers = {"Authorization": f"Bearer {lm_key}"} if lm_key else {}
            r = c.get(f"{lm_url.rstrip('/v1')}/v1/models", headers=headers)
            lm_ok = r.status_code == 200
            lm_info = f"reachable ({r.status_code})" if lm_ok else f"unreachable ({r.status_code})"
    except Exception as exc:
        lm_ok = False
        lm_info = f"unreachable ({exc})"
    add_check("LM Studio", lm_info, lm_ok)

    lm_models: set[str] = set()
    def _load_lmstudio_models() -> set[str]:
        """Best-effort view of the loaded LM Studio model ids."""
        nonlocal lm_models
        if lm_models or not lm_ok:
            return lm_models
        try:
            import httpx
            headers = {"Authorization": f"Bearer {lm_key}"} if lm_key else {}
            with httpx.Client(timeout=2.0) as c:
                r = c.get(f"{lm_url.rstrip('/v1')}/v1/models", headers=headers)
            if r.status_code != 200:
                return lm_models
            payload = r.json()
        except Exception:
            return lm_models
        raw_models = []
        if isinstance(payload, dict):
            raw_models = payload.get("data") or payload.get("models") or []
        elif isinstance(payload, list):
            raw_models = payload
        for item in raw_models:
            if isinstance(item, dict):
                model_id = item.get("id") or item.get("name") or item.get("model")
            else:
                model_id = item
            if model_id:
                lm_models.add(str(model_id))
        return lm_models

    # 5. Decide whether any embedding provider path is actually usable.
    try:
        importlib.import_module("sentence_transformers")
        local_ok = True
    except ImportError:
        local_ok = False

    if explicit_provider == "openai":
        provider_ok = openai_key
        provider_label = "openai" if provider_ok else "openai (missing OPENAI_API_KEY)"
    elif explicit_provider == "minimax":
        provider_ok = minimax_key
        provider_label = "minimax" if provider_ok else "minimax (missing MINIMAX_API_KEY)"
    elif explicit_provider == "lmstudio":
        provider_ok = lm_ok
        provider_label = "lmstudio" if provider_ok else "lmstudio (unreachable)"
    elif explicit_provider == "local":
        provider_ok = local_ok
        provider_label = "local" if provider_ok else "local (sentence_transformers missing)"
    else:
        if lm_ok:
            provider_ok = True
            provider_label = "lmstudio"
        elif minimax_key:
            provider_ok = True
            provider_label = "minimax"
        elif openai_key:
            provider_ok = True
            provider_label = "openai"
        elif local_ok:
            provider_ok = True
            provider_label = "local"
        else:
            provider_ok = False
            provider_label = "none"
    add_check("embedding provider", provider_label, provider_ok)

    # 6. Store reachable (default dim)
    try:
        store, emb = await _resolve_store_and_embedder()
        stats = store.stats()
        tiers_with_data = sum(1 for t in ['working', 'episodic', 'semantic', 'procedural'] if getattr(stats, t, 0) > 0)
        # emb may be None if no provider is available (all detection paths failed).
        prov_name = getattr(emb, "name", "unconfigured") or "unconfigured"
        prov_dim = getattr(emb, "dim", 1536) or 1536
        add_check("chroma store", f"{stats.total} chunks across {tiers_with_data} tiers (provider={prov_name}, dim={prov_dim})", True)
        if prov_name == "lmstudio":
            models = _load_lmstudio_models()
            embed_model = os.environ.get("LMSTUDIO_MODEL", "text-embedding-embeddinggemma-300m")
            embed_ok = embed_model in models
            add_check("LM Studio embedding model", embed_model if embed_ok else f"missing ({embed_model})", embed_ok)
            required_names.add("LM Studio embedding model")
            if os.environ.get("DUCKBOT_RERANK", "0").lower() in ("1", "true", "yes") or os.environ.get("LMSTUDIO_RERANK_URL"):
                rerank_model = os.environ.get("LMSTUDIO_RERANK_MODEL", "qwen3-reranker-0.6b")
                rerank_ok = rerank_model in models
                add_check("LM Studio reranker model", rerank_model if rerank_ok else f"missing ({rerank_model})", rerank_ok)
                required_names.add("LM Studio reranker model")
    except Exception as exc:
        add_check("chroma store", str(exc), False)

    if explicit_provider == "lmstudio":
        required_names.add("LM Studio")

    all_ok = all(ok for name, _, ok in checks if name in required_names)
    return checks, all_ok


def cmd_doctor(args: argparse.Namespace) -> int:
    """Sanity check: env, deps, store."""
    checks, all_ok = asyncio.run(build_doctor_checks_async())
    if getattr(args, "json", False):
        print(json.dumps({
            "ok": all_ok,
            "checks": [
                {"name": name, "value": value, "ok": ok}
                for name, value, ok in checks
            ],
        }, indent=2, default=str))
        return 0 if all_ok else 1

    max_name = max(len(c[0]) for c in checks)
    for name, value, ok in checks:
        marker = "✓" if ok else "✗"
        print(f"  {marker} {name.ljust(max_name)}  {value}")
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


def cmd_openclaw(args: argparse.Namespace) -> int:
    """OpenClaw CLI shim: 'python -m src.cli openclaw <verb> [args...]' delegates to the shared 12-tool surface."""
    from .connectors import openclaw_shim
    return openclaw_shim.main(list(args.verb))


def cmd_skills(args: argparse.Namespace) -> int:
    """Skill pipeline CLI: 'python -m src.cli skills <verb> [args...]'.

    Storage-only — no LLM. Agents use this to inspect and promote
    skill candidates they previously stamped.

    Verbs:
      stamp <text>            - stamp a new skill candidate
      list [-k N] [--include-promoted]  - list unpromoted candidates
      promote <chunk_id> <name> <description> <instr1> [instr2 ...]
      promote <chunk_id> --json '<json-args>'
      suggest <query> [-k N]  - semantic top-N candidates matching a query
    """
    import json as _json
    from .skill_pipeline import stamp_skill_candidate, list_candidates, promote_candidate, suggest_candidates
    from .connectors.base import _run_async

    verb = args.verb[0]
    rest = args.verb[1:]

    if verb in ("stamp",):
        if not rest:
            print('{"error": "skills stamp requires <text>"}')
            return 2
        text = " ".join(rest)
        result = stamp_skill_candidate(text=text)
        print(json.dumps({
            "chunk_id": result.chunk_id,
            "tier": result.tier,
            "stored": result.stored,
        }, indent=2, default=str))
        return 0 if result.stored else 2

    if verb in ("list",):
        args_dict = {}
        if "--include-promoted" in rest:
            args_dict["include_promoted"] = True
        if "-k" in rest:
            idx = rest.index("-k")
            if idx + 1 < len(rest):
                try:
                    args_dict["k"] = int(rest[idx + 1])
                except ValueError:
                    pass
        out = list_candidates(**args_dict)
        if isinstance(out, dict) and out.get("error"):
            print(json.dumps(out))
            return 2
        print(json.dumps({"candidates": out}, indent=2, default=str))
        return 0

    if verb in ("promote",):
        # JSON form: skills promote <chunk_id> --json '<json-args>'
        if "--json" in rest:
            idx = rest.index("--json")
            chunk_id = rest[0] if idx > 0 else ""
            raw = " ".join(rest[idx + 1:]).strip()
            if not chunk_id or not raw:
                print('{"error": "skills promote --json requires <chunk_id> and json-args"}')
                return 2
            try:
                parsed = _json.loads(raw)
            except _json.JSONDecodeError as e:
                print(json.dumps({"error": f"json-args not valid JSON: {e}"}))
                return 2
            if not isinstance(parsed, dict):
                print(json.dumps({"error": "json-args must be a JSON object"}))
                return 2
            parsed["chunk_id"] = chunk_id
            out = promote_candidate(**parsed)
        else:
            # Positional: <chunk_id> <name> <description> <instr1> [instr2 ...]
            if len(rest) < 4:
                print('{"error": "positional form requires: skills promote <chunk_id> <name> <description> <instr1> [instr2 ...]"}')
                return 2
            chunk_id, name, description, *instructions = rest
            out = promote_candidate(
                chunk_id=chunk_id,
                name=name,
                description=description,
                instructions=instructions,
            )
        print(json.dumps(out, indent=2, default=str))
        return 0 if out.get("promoted") else 2

    if verb in ("suggest",):
        if not rest:
            print('{"error": "skills suggest requires <query>"}')
            return 2
        args_dict = {"query": " ".join(rest)}
        if "-k" in rest:
            idx = rest.index("-k")
            if idx + 1 < len(rest):
                try:
                    args_dict["k"] = int(rest[idx + 1])
                except ValueError:
                    pass
        out = suggest_candidates(**args_dict)
        print(json.dumps({"candidates": out}, indent=2, default=str))
        return 0

    print(json.dumps({"error": f"unknown skills verb: {verb}", "available": ["stamp", "list", "promote", "suggest"]}))
    return 1


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

    # ---- Maintenance (v0.15.2) --------------------------------------------
    p_fsck = sub.add_parser("fsck", help="per-tier health report — disk size, "
                                          "vector count, HNSW bloat check")
    p_fsck.set_defaults(func=cmd_fsck)

    p_vacuum = sub.add_parser("vacuum", help="drop a single tier's "
                                              "ChromaDB collection (frees disk immediately)")
    p_vacuum.add_argument("tier", help="tier to vacuum (working/episodic/"
                                          "semantic/procedural)")
    p_vacuum.add_argument("--yes", action="store_true", help="confirm")
    p_vacuum.set_defaults(func=cmd_vacuum)

    p_reindex = sub.add_parser("reindex-tier", help="wipe a tier and re-ingest "
                                                      "from watcher state (use after vacuum)")
    p_reindex.add_argument("tier", help="tier to reindex")
    p_reindex.add_argument("--yes", action="store_true", help="confirm")
    p_reindex.add_argument("--chunk-size", type=int, default=512)
    p_reindex.add_argument("--overlap", type=float, default=0.15)
    p_reindex.set_defaults(func=cmd_reindex_tier)

    p_prune = sub.add_parser("prune-empty-collections",
                              help="delete empty non-tier Chroma collections")
    p_prune.add_argument("--yes", action="store_true", help="confirm")
    p_prune.set_defaults(func=cmd_prune_empty_collections)

    p_purge = sub.add_parser("purge-quarantine",
                              help="delete old quarantined items")
    p_purge.add_argument("--older-than-days", type=int, default=30,
                          help="delete items added more than N days ago (default 30)")
    p_purge.set_defaults(func=cmd_purge_quarantine)

    p_rotate = sub.add_parser("rotate-events",
                                help="rotate data/events.db when it exceeds a size cap")
    p_rotate.add_argument("--max-mb", type=int, default=50,
                            help="rotate when events.db > N MB (default 50)")
    p_rotate.add_argument("--gzip", action="store_true", default=True,
                            help="gzip-compress the rotated file (default true)")
    p_rotate.set_defaults(func=cmd_rotate_events)

    p_maint = sub.add_parser("maintenance",
                                help="run a safe battery of cleanups in one shot "
                                      "(prune-empty-collections + purge-quarantine + "
                                      "rotate-events + fsck)")
    p_maint.set_defaults(func=cmd_maintenance)

    p_doc = sub.add_parser("doctor", help="check env + deps")
    p_doc.add_argument("--json", action="store_true", help="output as JSON")
    p_doc.set_defaults(func=cmd_doctor)

    # Update: pull latest from origin/main + upgrade deps + doctor.
    p_update = sub.add_parser("update", help="pull latest from origin/main and upgrade deps")
    p_update.add_argument(
        "--dry-run", "--check", dest="dry_run", action="store_true",
        help="only check whether updates are available without pulling",
    )
    p_update.add_argument(
        "--no-deps", action="store_true",
        help="skip pip install and requirements.txt upgrade",
    )
    p_update.add_argument(
        "--no-doctor", action="store_true",
        help="skip the doctor check after update",
    )
    p_update.set_defaults(func=cmd_update)

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

    p_openclaw = sub.add_parser("openclaw", help="OpenClaw agent CLI shim: openclaw <verb> [args...]")
    p_openclaw.add_argument("verb", nargs="+", help="verb (wake-up, recall, remember, stats, tools, call, etc.) + args")
    p_openclaw.set_defaults(func=cmd_openclaw)

    # Skills pipeline: stamp / list / promote / suggest (no LLM).
    # Use argparse.REMAINDER for the verb+args so the subcommand can parse
    # its own flags (e.g. skills list -k 3) without argparse rejecting
    # them as "unrecognized arguments".
    p_skills = sub.add_parser("skills", help="agent-driven skill pipeline: skills <verb> [args...]")
    p_skills.add_argument("verb", nargs=argparse.REMAINDER, help="verb (list / promote / suggest / stamp) + args")
    p_skills.set_defaults(func=cmd_skills)

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
