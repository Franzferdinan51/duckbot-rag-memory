"""
watcher.py — automatic memory update daemon.

Watches a directory tree for new/changed/deleted markdown files and syncs them
into the memory store. The auto-update pattern from mem0's hook system,
adapted for filesystem events.

This is the replacement for cron-based batch ingestion. Per Duckets (2026-06-23):
  - No automatic cron
  - Memory should update in real time
  - "Do it however others do it"

How others do it (research 2026-06-23):
  - mem0: hooks on session events call add()/update()
  - Letta: persistent auto-save on every message
  - Cognee: add() is the canonical entry; cognify() is opt-in batch
  - Hermes Agent: FTS5 + periodic nudge

We combine: filesystem-event triggers (inotify/FSEvents) + add() on every change.
Result: latency is seconds, not hours. Cost is the same (only changed chunks
re-embed, and we have content-hash dedup so unchanged content is free).

Usage:
    python -m src.watcher /path/to/watch                 # foreground
    python -m src.watcher --daemon /path/to/watch        # daemonize
    python -m src.watcher --status                       # is the daemon running?
    python -m src.watcher --stop                         # kill the daemon
    python -m src.watcher --once /path/to/watch          # run one pass, exit
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path

from .chunk import iter_markdown_files
from .memory import Memory
from .store import MemoryStore
from .tier import Tier


# State file: tracks which file paths we've seen + their mtime + chunk ids
STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "watcher_state.json"
LOG_PATH = Path(__file__).resolve().parent.parent / "data" / "watcher.log"
PID_PATH = Path(__file__).resolve().parent.parent / "data" / "watcher.pid"
DEFAULT_WATCH = [
    Path.home() / ".openclaw" / "workspace" / "memory",
    Path.home() / ".openclaw" / "workspace" / "MEMORY.md",
    Path.home() / ".openclaw" / "workspace" / "AGENTS.md",
    Path.home() / ".openclaw" / "workspace" / "SOUL.md",
    Path.home() / ".openclaw" / "workspace" / "IDENTITY.md",
    Path.home() / "Desktop" / "ai-Py-boy-emulation-main",
    Path.home() / "Desktop" / "Newest Desktop Control",
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S%z')}] {msg}"
    print(line)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            pass
    return {"files": {}, "last_run": 0.0, "total_remembered": 0, "total_forgotten": 0}


def save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


async def sync_files(paths: list[str], state: dict) -> dict:
    """One full sync pass. Returns stats: {added, updated, deleted, skipped, errors}."""
    mem = Memory()
    store, _ = await mem._ensure_initialized()
    stats = {"added": 0, "updated": 0, "deleted": 0, "skipped": 0, "errors": []}

    # Build the set of current files
    current_files: dict[str, float] = {}  # path -> mtime
    for source, contents in iter_markdown_files(paths):
        if not source:
            continue
        try:
            mtime = os.path.getmtime(source)
        except OSError:
            continue
        current_files[source] = mtime

    # 1. Handle deletes (files in state but no longer present)
    known_files = set(state.get("files", {}).keys())
    for path in known_files - set(current_files.keys()):
        chunk_ids = state["files"][path].get("chunk_ids", [])
        for cid in chunk_ids:
            for tier in Tier:
                try:
                    store.collection_for(tier).delete(ids=[cid])
                except Exception:
                    pass
        log(f"  deleted {len(chunk_ids)} chunks from {path}")
        stats["deleted"] += len(chunk_ids)
        del state["files"][path]

    # 2. Handle new/changed files
    for path, mtime in current_files.items():
        prev = state["files"].get(path, {})
        prev_mtime = prev.get("mtime", 0.0)
        if mtime <= prev_mtime:
            stats["skipped"] += 1
            continue

        # Read file
        try:
            content = Path(path).read_text(encoding="utf-8", errors="ignore")
        except Exception as exc:
            stats["errors"].append(f"read {path}: {exc}")
            continue

        # Delete prior chunks for this file
        old_chunk_ids = prev.get("chunk_ids", [])
        for cid in old_chunk_ids:
            for tier in Tier:
                try:
                    store.collection_for(tier).delete(ids=[cid])
                except Exception:
                    pass

        # Chunk + ingest via the new remember() pipeline
        from .chunk import chunk_markdown
        from .tier import classify, reclassify_for_working
        chunks = chunk_markdown(content, source_path=path, chunk_size=512)
        new_chunk_ids = []
        for c in chunks:
            try:
                a = classify(c.source_path, c.text)
                a = reclassify_for_working(c.source_path, a)
                r = await mem.remember(
                    c.text, source_path=c.source_path, metadata={"section_header": c.section_header}
                )
                new_chunk_ids.append(r.chunk_id)
            except Exception as exc:
                stats["errors"].append(f"remember {path}:{c.chunk_index}: {exc}")

        state["files"][path] = {
            "mtime": mtime,
            "chunk_ids": new_chunk_ids,
            "last_sync": time.time(),
            "chunk_count": len(new_chunk_ids),
        }
        if prev_mtime > 0:
            stats["updated"] += len(new_chunk_ids)
            log(f"  updated {len(new_chunk_ids)} chunks from {path}")
        else:
            stats["added"] += len(new_chunk_ids)
            log(f"  added {len(new_chunk_ids)} chunks from {path}")

    state["last_run"] = time.time()
    state["total_remembered"] = state.get("total_remembered", 0) + stats["added"] + stats["updated"]
    state["total_forgotten"] = state.get("total_forgotten", 0) + stats["deleted"]
    save_state(state)
    return stats


# -----------------------------------------------------------------------------
# Filesystem watcher (uses watchdog if available, falls back to polling)
# -----------------------------------------------------------------------------

class PollingHandler:
    """Simple polling handler — works without any external deps.

    Every poll_interval seconds, runs a sync pass.
    """
    def __init__(self, paths: list[str], interval: float = 2.0):
        self.paths = paths
        self.interval = interval
        self.state = load_state()
        self._stop = False

    def stop(self):
        self._stop = True

    async def run(self):
        log(f"Polling watcher starting on {len(self.paths)} paths (interval={self.interval}s)")
        while not self._stop:
            try:
                stats = await sync_files(self.paths, self.state)
                if stats["added"] or stats["updated"] or stats["deleted"]:
                    log(f"sync pass: {stats}")
            except Exception as exc:
                log(f"sync error: {exc}")
            await asyncio.sleep(self.interval)
        log("Polling watcher stopped")


def start_watchdog_handler(paths: list[str], interval: float = 2.0):
    """Try to use watchdog (FSEvents on macOS, inotify on Linux). Falls back to polling."""
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler

        class MDHandler(FileSystemEventHandler):
            def __init__(self, state: dict, paths: list[str]):
                self.state = state
                self.paths = paths
                self._pending = False

            def _maybe_trigger(self, event_path: str):
                if event_path.endswith(".md"):
                    log(f"  event: {event_path}")
                    # Coalesce events: schedule a sync
                    self._pending = True

            def on_created(self, event):
                if not event.is_directory:
                    self._maybe_trigger(event.src_path)

            def on_modified(self, event):
                if not event.is_directory:
                    self._maybe_trigger(event.src_path)

            def on_deleted(self, event):
                if not event.is_directory:
                    self._maybe_trigger(event.src_path)

            def on_moved(self, event):
                if not event.is_directory:
                    self._maybe_trigger(event.dest_path)

        state = load_state()
        handler = MDHandler(state, paths)
        observer = Observer()
        for p in paths:
            pp = Path(p)
            if pp.is_file():
                pp = pp.parent
            if pp.exists():
                observer.schedule(handler, str(pp), recursive=True)
                log(f"  watching {pp}")
        observer.start()

        async def loop():
            try:
                while True:
                    await asyncio.sleep(interval)
                    if handler._pending:
                        handler._pending = False
                        try:
                            await sync_files(paths, state)
                        except Exception as exc:
                            log(f"sync error: {exc}")
            finally:
                observer.stop()
                observer.join()

        return asyncio.run(loop())
    except ImportError:
        log("watchdog not installed, using polling")
        return PollingHandler(paths, interval=interval).run()


def write_pid() -> None:
    PID_PATH.write_text(str(os.getpid()))


def clear_pid() -> None:
    if PID_PATH.exists():
        PID_PATH.unlink()


def cmd_status() -> int:
    if not PID_PATH.exists():
        print("Watcher: not running")
        return 0
    pid = int(PID_PATH.read_text().strip())
    try:
        os.kill(pid, 0)
        print(f"Watcher: running (pid={pid})")
    except ProcessLookupError:
        print(f"Watcher: stale pid file (pid={pid} not alive)")
        PID_PATH.unlink()
    return 0


def cmd_stop() -> int:
    if not PID_PATH.exists():
        print("Watcher: not running")
        return 0
    pid = int(PID_PATH.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Watcher: sent SIGTERM to pid={pid}")
    except ProcessLookupError:
        print(f"Watcher: pid={pid} not alive")
    PID_PATH.unlink()
    return 0


def cmd_once(args) -> int:
    paths = args.paths or [str(p) for p in DEFAULT_WATCH]
    state = load_state()
    stats = asyncio.run(sync_files(paths, state))
    print(json.dumps(stats, indent=2))
    return 0


def cmd_run(args) -> int:
    """Run the watcher in foreground."""
    paths = args.paths or [str(p) for p in DEFAULT_WATCH]
    write_pid()
    try:
        start_watchdog_handler(paths, interval=args.interval)
    finally:
        clear_pid()
    return 0


def cmd_daemon(args) -> int:
    """Daemonize: fork, write pid, return."""
    if PID_PATH.exists():
        pid = int(PID_PATH.read_text().strip())
        try:
            os.kill(pid, 0)
            print(f"Watcher already running (pid={pid})")
            return 1
        except ProcessLookupError:
            PID_PATH.unlink()
    pid = os.fork()
    if pid > 0:
        # Parent
        print(f"Watcher daemonized: pid={pid}")
        return 0
    # Child
    os.setsid()
    pid2 = os.fork()
    if pid2 > 0:
        os._exit(0)
    # Grandchild
    sys.stdin.close()
    sys.stdout.close()
    sys.stderr.close()
    with open("/dev/null", "r") as devnull:
        os.dup2(devnull.fileno(), 0)
    with open(str(LOG_PATH), "a+") as logf:
        os.dup2(logf.fileno(), 1)
        os.dup2(logf.fileno(), 2)
    write_pid()
    paths = args.paths or [str(p) for p in DEFAULT_WATCH]
    try:
        start_watchdog_handler(paths, interval=args.interval)
    finally:
        clear_pid()
    return 0


def main():
    p = argparse.ArgumentParser(prog="duckbot-watcher", description="Automatic memory update daemon")
    sub = p.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="run in foreground")
    p_run.add_argument("paths", nargs="*", help="paths to watch (files or directories)")
    p_run.add_argument("--interval", type=float, default=2.0, help="poll interval seconds")
    p_run.set_defaults(func=cmd_run)

    p_d = sub.add_parser("daemon", help="daemonize (fork into background)")
    p_d.add_argument("paths", nargs="*", help="paths to watch")
    p_d.add_argument("--interval", type=float, default=2.0, help="poll interval seconds")
    p_d.set_defaults(func=cmd_daemon)

    p_o = sub.add_parser("once", help="run one sync pass and exit")
    p_o.add_argument("paths", nargs="*", help="paths to watch")
    p_o.set_defaults(func=cmd_once)

    p_s = sub.add_parser("status", help="check if daemon is running")
    p_s.set_defaults(func=lambda a: cmd_status())

    p_x = sub.add_parser("stop", help="stop the daemon")
    p_x.set_defaults(func=lambda a: cmd_stop())

    args = p.parse_args()
    if not hasattr(args, "func"):
        p.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
