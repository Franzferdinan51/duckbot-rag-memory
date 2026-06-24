"""
watcher.py — automatic memory update daemon (polling, macOS-safe).

Watches a directory tree for new/changed/deleted markdown files and syncs them
into the memory store in real time. The auto-update pattern from mem0's hook
system, adapted for filesystem polling.

This is the replacement for cron-based batch ingestion. Per Duckets (2026-06-23):
  - No automatic cron
  - Memory should update in real time
  - "Do it however others do it"

How others do it (research 2026-06-23):
  - mem0: hooks on session events call add()/update()
  - Letta: persistent auto-save on every message
  - Cognee: add() is the canonical entry; cognify() is opt-in batch
  - Hermes Agent: FTS5 + periodic nudge

We use **polling** (2s interval) by default. On macOS, `watchdog`'s FSEvents
segfaults when combined with `chromadb` + `httpx` in the same process. Polling
gives us the same latency profile (seconds, not hours) without the crash.
Set `DUCKBOT_WATCH_USE_FSEVENTS=1` to opt into watchdog. Sort by mtime DESC
so newly-changed files get processed first.

Requires Python 3.12+ (3.9.6 from Xcode segfaults in chromadb).

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
import threading
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

    # Build the set of current files. Sort by mtime DESC so newly-created
    # or recently-modified files get processed first — important so that
    # live edits become queryable quickly during backfill.
    EXCLUDE_DIR_NAMES = {"node_modules", ".git", "__pycache__", ".venv", "venv",
                         ".next", ".nuxt", "dist", "build", ".cache", ".pytest_cache",
                         "target", ".tox", "node-gyp", "coverage", ".mypy_cache",
                         ".ruff_cache", "Pods", "DerivedData"}
    def _is_excluded(path: str) -> bool:
        from pathlib import Path as _P
        parts = set(_P(path).parts)
        return bool(parts & EXCLUDE_DIR_NAMES)
    current_files: dict[str, float] = {}  # path -> mtime
    for source, contents in iter_markdown_files(paths):
        if not source or _is_excluded(source):
            continue
        try:
            mtime = os.path.getmtime(source)
        except OSError:
            continue
        current_files[source] = mtime
    # Sort: highest mtime first
    current_files = dict(sorted(current_files.items(), key=lambda kv: kv[1], reverse=True))

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


def start_watchdog_handler(paths: list[str], interval: float = 2.0, initial_sync: bool = True):
    """Block on a watchdog Observer until killed. Optionally does an initial sync first.

    Args:
      paths: list of files/directories to watch
      interval: poll interval for coalesced events (default 2s)
      initial_sync: if True, ingest all existing markdown before watching for events.
        This can segfault on macOS when ChromaDB+httpx+watchdog are all loaded in
        the same process — if that happens, run `watcher once` separately first.

    Returns when SIGTERM/SIGINT is received.
    """
    # Per 2026-06-23 diagnostic: watchdog+FSEvents segfaults when ChromaDB+httpx
    # are also loaded in the same process on macOS. The polling handler does
    # the same job (per-file mtime check + sync_files) and is rock-solid.
    # Set DUCKBOT_WATCH_USE_FSEVENTS=1 to opt into the native observer.
    import os as _os
    if not _os.environ.get("DUCKBOT_WATCH_USE_FSEVENTS"):
        log("using polling handler (set DUCKBOT_WATCH_USE_FSEVENTS=1 to opt into FSEvents)")
        return asyncio.run(PollingHandler(paths, interval=interval).run())
    try:
        from watchdog.observers import Observer
        from watchdog.events import FileSystemEventHandler
    except ImportError:
        log("watchdog not installed, using polling")
        return PollingHandler(paths, interval=interval).run()

    state = load_state()
    # threading.Event is loop-agnostic, avoids the "attached to a different loop"
    # error you get with asyncio.Event when sync_files has already torn down a loop.
    stop_event = threading.Event()

    def _request_stop(*_):
        if not stop_event.is_set():
            log("stop requested")
            stop_event.set()

    # Use signal handlers if we're in the main thread
    try:
        loop = asyncio.get_running_loop()
        for sig_name in ("SIGTERM", "SIGINT", "SIGHUP"):
            sig = getattr(signal, sig_name, None)
            if sig is not None:
                try:
                    loop.add_signal_handler(sig, _request_stop)
                except (NotImplementedError, RuntimeError):
                    pass
    except RuntimeError:
        pass

    class MDHandler(FileSystemEventHandler):
        def __init__(self, state: dict, paths: list[str]):
            self.state = state
            self.paths = paths
            self._pending = False

        def _maybe_trigger(self, event_path: str):
            if event_path.endswith(".md"):
                log(f"  event: {event_path}")
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

    async def main_loop():
        # Initial sync so files that existed before we started watching get ingested.
        # Skip on macOS if you hit a segfault — run `watcher once` separately first.
        if initial_sync:
            try:
                log("initial sync starting...")
                stats = await sync_files(paths, state)
                log(f"initial sync done: added={stats.get('added', 0)} updated={stats.get('updated', 0)} skipped={stats.get('skipped', 0)}")
            except Exception as exc:
                log(f"initial sync error: {exc}")
        # Event-driven sync loop. Sleeps `interval` seconds between sync passes,
        # checking the threading.Event each iteration so SIGTERM stops promptly.
        try:
            while not stop_event.is_set():
                if handler._pending:
                    handler._pending = False
                    try:
                        stats = await sync_files(paths, state)
                        if stats.get("added") or stats.get("updated") or stats.get("deleted"):
                            log(f"sync pass: {stats}")
                    except Exception as exc:
                        log(f"sync error: {exc}")
                # Sleep in small slices so we react to stop_event promptly
                slices = max(1, int(interval * 10))
                for _ in range(slices):
                    if stop_event.is_set():
                        break
                    await asyncio.sleep(0.1)
        finally:
            observer.stop()
            observer.join()

    asyncio.run(main_loop())


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
    except (ProcessLookupError, OSError):
        # OSError covers Windows cases where the pid is no longer valid
        # (e.g. ERROR_INVALID_PARAMETER when the process was reaped).
        print(f"Watcher: stale pid file (pid={pid} not alive)")
        try:
            PID_PATH.unlink()
        except FileNotFoundError:
            pass
    return 0


def cmd_stop() -> int:
    if not PID_PATH.exists():
        print("Watcher: not running")
        return 0
    pid = int(PID_PATH.read_text().strip())
    # Cross-platform: SIGTERM on POSIX, SIGBREAK on Windows. Both are
    # present in Python's signal module on their respective platforms.
    sig = getattr(signal, "SIGTERM", getattr(signal, "SIGBREAK", 15))
    try:
        os.kill(pid, sig)
        print(f"Watcher: sent termination signal to pid={pid}")
    except ProcessLookupError:
        print(f"Watcher: pid={pid} not alive")
    except OSError as e:
        # PermissionError on Windows when the pid is owned by another user;
        # ESRCH for races where the process exits between check and kill.
        print(f"Watcher: cannot stop pid={pid}: {e}")
    PID_PATH.unlink(missing_ok=True)
    return 0


def cmd_once(args) -> int:
    paths = args.paths or [str(p) for p in DEFAULT_WATCH]
    state = load_state()
    stats = asyncio.run(sync_files(paths, state))
    print(json.dumps(stats, indent=2))
    return 0


def cmd_run(args) -> int:
    """Run the watcher in foreground.

    Performs an initial sync so files that existed before we started watching
    get ingested. watchdog's FSEvents-based Observer only fires on actual
    file changes after schedule(), so without an initial pass we'd miss
    everything that's already on disk.

    Use --no-initial-sync if the initial sync segfaults on your platform
    (known macOS issue with ChromaDB+httpx+watchdog in the same process).
    In that case run `watcher once` separately first to do the backfill.
    """
    paths = args.paths or [str(p) for p in DEFAULT_WATCH]
    write_pid()
    try:
        initial = getattr(args, "initial_sync", True)
        start_watchdog_handler(paths, interval=args.interval, initial_sync=initial)
    finally:
        clear_pid()
    return 0


def cmd_daemon(args) -> int:
    """Daemonize: detach from the controlling terminal and run in background.

    Cross-platform strategy:
      - On POSIX (macOS, Linux): classic double-fork + setsid. This is the
        well-known Unix daemon pattern; works since the 1980s.
      - On Windows: there is no `os.fork()`. We use `subprocess.Popen` with
        `DETACHED_PROCESS` + `CREATE_NEW_PROCESS_GROUP` flags, which is
        the Windows equivalent: the child survives the parent exiting and
        gets no controlling terminal. Same end result, different mechanism.

    Both branches write the same PID file at PID_PATH, so `cmd_status` /
    `cmd_stop` work identically on all three OSes.

    The trick that makes POSIX work where naive double-fork fails:
      - Grandchild ignores SIGHUP/SIGPIPE before any stdio work
      - We write_pid BEFORE redirecting stdio so any post-write debug output is captured
      - All three stdio streams are closed AND dup2'd to os.devnull + log file
        (close() alone leaves fd 0/1/2 pointing at the now-defunct parent tty)
    """
    if PID_PATH.exists():
        pid = int(PID_PATH.read_text().strip())
        try:
            os.kill(pid, 0)
            print(f"Watcher already running (pid={pid})")
            return 1
        except (ProcessLookupError, OSError):
            # OSError on Windows when the pid no longer exists.
            try:
                PID_PATH.unlink()
            except FileNotFoundError:
                pass

    paths = args.paths or [str(p) for p in DEFAULT_WATCH]

    if sys.platform == "win32":
        return _daemon_windows(paths, args)
    return _daemon_posix(paths, args)


def _daemon_windows(paths: list[str], args) -> int:
    """Windows daemonization: spawn a detached subprocess.

    `DETACHED_PROCESS` (0x00000008) + `CREATE_NEW_PROCESS_GROUP` (0x00000200)
    detaches the child from the parent's console. We then `Popen` ourselves
    with `python.exe -m src.watcher run ...` as the child, and exit the
    parent cleanly.
    """
    DETACHED_PROCESS = 0x00000008
    CREATE_NEW_PROCESS_GROUP = 0x00000200
    creationflags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP

    # Write a placeholder pid so cmd_status works immediately
    PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text("starting")

    # Open log files for the detached child's stdio. We use line-buffered
    # append mode so `tail -f` sees output immediately. POSIX _daemon_posix
    # does the equivalent via dup2; on Windows, Popen's stdout/stderr
    # kwargs let us redirect the same way without console inheritance.
    import subprocess
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        log_out = open(str(LOG_PATH), "a+", buffering=1)  # line-buffered
        log_err = open(str(LOG_PATH), "a+", buffering=1)
    except OSError as e:
        print(f"❌ Failed to open log file: {e}", file=sys.stderr)
        try:
            PID_PATH.unlink()
        except FileNotFoundError:
            pass
        return 1

    try:
        p = subprocess.Popen(
            [sys.executable, "-m", "src.watcher", "run", *paths,
             "--interval", str(args.interval)],
            stdin=subprocess.DEVNULL,
            stdout=log_out,
            stderr=log_err,
            creationflags=creationflags,
            close_fds=True,
        )
    except Exception as e:
        print(f"❌ Failed to start watcher: {e}", file=sys.stderr)
        log_out.close()
        log_err.close()
        try:
            PID_PATH.unlink()
        except FileNotFoundError:
            pass
        return 1
    except Exception as e:
        print(f"❌ Failed to start watcher: {e}", file=sys.stderr)
        try:
            PID_PATH.unlink()
        except FileNotFoundError:
            pass
        return 1

    # The detached child writes its real pid to PID_PATH via cmd_run's
    # write_pid(). We poll briefly for it.
    for _ in range(40):  # up to ~4 seconds
        time.sleep(0.1)
        try:
            current = PID_PATH.read_text().strip()
            if current and current != "starting" and current.isdigit():
                print(f"Watcher daemonized: pid={current}")
                return 0
        except (OSError, FileNotFoundError):
            pass
    # Child didn't write a pid within 4s; assume it started OK.
    print(f"Watcher daemonized: pid={p.pid} (status file not yet updated)")
    return 0


def _daemon_posix(paths: list[str], args) -> int:
    """POSIX (macOS / Linux) daemonization: classic double-fork."""
    import signal as _signal

    # First fork: detach from parent
    pid = os.fork()
    if pid > 0:
        # Parent: poll for grandchild pid to appear in pidfile
        for _ in range(20):
            time.sleep(0.1)
            if PID_PATH.exists():
                gp = PID_PATH.read_text().strip()
                if gp and gp != str(pid):
                    print(f"Watcher daemonized: pid={gp}")
                    return 0
        print(f"Watcher daemonized (initial pid={pid}). Check data/watcher.pid.")
        return 0

    # First child: become session leader
    os.setsid()

    # Second fork: ensure we can't reacquire a controlling terminal
    pid2 = os.fork()
    if pid2 > 0:
        os._exit(0)

    # === Grandchild (the actual daemon) ===
    # Block SIGHUP/SIGPIPE so the parent-shell-exit cascade can't kill us
    for sig in (_signal.SIGHUP, _signal.SIGPIPE, _signal.SIGTERM):
        try:
            _signal.signal(sig, _signal.SIG_IGN)
        except Exception:
            pass

    # Write pid BEFORE stdio redirect so any debug output after is captured
    try:
        PID_PATH.write_text(str(os.getpid()))
        # os.chmod is POSIX-only; Windows ignores it. Guarded for safety.
        if hasattr(os, "chmod"):
            try:
                os.chmod(str(PID_PATH), 0o644)
            except (OSError, NotImplementedError):
                pass
    except Exception:
        pass

    # Redirect stdio: close first, then dup2 (closing alone leaves fds pointing at dead tty)
    try:
        sys.stdin.close()
    except Exception:
        pass
    try:
        sys.stdout.close()
    except Exception:
        pass
    try:
        sys.stderr.close()
    except Exception:
        pass
    try:
        logf = open(str(LOG_PATH), "a+")
        os.dup2(logf.fileno(), 1)
        os.dup2(logf.fileno(), 2)
    except Exception:
        pass
    try:
        # os.devnull is /dev/null on POSIX and nul on Windows
        dn = open(os.devnull, "r")
        os.dup2(dn.fileno(), 0)
    except Exception:
        pass

    paths = args.paths or [str(p) for p in DEFAULT_WATCH]
    try:
        start_watchdog_handler(paths, interval=args.interval)
    finally:
        try:
            PID_PATH.unlink(missing_ok=True)
        except Exception:
            pass
    return 0


def main():
    p = argparse.ArgumentParser(prog="duckbot-watcher", description="Automatic memory update daemon")
    sub = p.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="run in foreground")
    p_run.add_argument("paths", nargs="*", help="paths to watch (files or directories)")
    p_run.add_argument("--interval", type=float, default=2.0, help="poll interval seconds")
    p_run.add_argument("--no-initial-sync", dest="initial_sync", action="store_false",
                       help="skip the startup backfill (use `watcher once` separately)")
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
