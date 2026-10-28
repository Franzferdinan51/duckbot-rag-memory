"""DuckBot brain — Hermes MemoryProvider plugin.

Implements Hermes's MemoryProvider ABC (NousResearch/hermes-agent,
`agent/memory_provider.py`). When this plugin is active, the agent
will:

  - Call `prefetch(query)` before each turn and inject relevant chunks.
  - Call `sync_turn(user, assistant)` after each turn to remember the
    conversation in the background (non-blocking).
  - Expose all 12 core brain tools (incl. `brain_wake_up`, the
    canonical session-start call) via `get_tool_schemas()`.
  - Run `on_session_end(messages)` to consolidate high-importance
    session facts into a durable procedural chunk.

This is the Hermes side of Layer 16 (cross-runtime integration).
Pattern source: NousResearch/hermes-agent/plugins/memory/{honcho,hindsight,holographic}.

v0.14.0: tool schemas + dispatch delegate to `src.extensions.tools` so
the surface stays in lock-step with the OpenClaw extension and any
future adapter. The plugin is now a thin Python adapter on top of the
shared core, plus the Hermes-specific lifecycle hooks (prefetch,
sync_turn, on_session_end).

No paid APIs. All local (LM Studio + Chroma + BGE reranker).
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

# Fix sys.path so `from src.connectors.base import Brain` resolves.
# When Hermes loads this plugin the cwd is ~ and Python has no
# duckbot-rag-memory on the path.  We locate the repo root using a
# two-pass strategy:
#
#  PASS 1 — expanduser("~") primary (Windows-safe, works on all platforms).
#    ~/duckbot-rag-memory is the canonical install location.
#
#  PASS 2 — upward walk from __file__ (dev / IDE layouts).
#    Handles the case where the repo IS above the plugin dir.
#
#  If both fail we raise ImportError with a useful message.
from pathlib import Path as _P
import os as _os

_DUCKBOT_ROOT: str | None = None

# PASS 1 — expanduser primary (handles Windows correctly; no .parent math)
_home = _P(_os.path.expanduser("~"))                    # ~  (C:\Users\franz on Windows)
_repo_home_sibling = _home / "duckbot-rag-memory"
if (_repo_home_sibling / "src" / "connectors" / "base.py").exists():
    _DUCKBOT_ROOT = str(_repo_home_sibling)

# PASS 2 — upward walk from __file__ (dev layout)
if _DUCKBOT_ROOT is None:
    for _up in range(8):
        _candidate = _P(__file__).resolve().parents[_up]
        if (_candidate / "src" / "connectors" / "base.py").exists():
            _DUCKBOT_ROOT = str(_candidate)
            break

if _DUCKBOT_ROOT is None:
    raise ImportError(
        "duckbot-brain plugin could not locate duckbot-rag-memory.\n"
        f"  Tried: {_repo_home_sibling}\n"
        f"  Walked from: {_P(__file__).resolve()}\n"
        "Expected to find src/connectors/base.py either at\n"
        "  ~/duckbot-rag-memory/ (expanduser path) or\n"
        "  somewhere above the plugin __init__.py location."
    )

if _DUCKBOT_ROOT not in sys.path:
    sys.path.insert(0, _DUCKBOT_ROOT)

from src.connectors.base import Brain
from src.extensions import tools as _tools

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Bridge mode (v0.16.0)
#
# The Hermes gateway runs an isolated bundled Python (-I: no user
# site-packages, PYTHONPATH stripped) that cannot import chromadb, so
# the in-process Brain can never run there — the 2026-10-19 host gate
# made the provider self-disable in exactly the process that serves
# agents. Bridge mode removes that dead end: when the in-process import
# fails, the provider spawns src.brain_worker under the repo venv
# Python (the same interpreter the duckbot-memory MCP server uses) and
# forwards Brain calls over stdio line-JSON. Same store, same code —
# the Brain simply lives in an interpreter that can host it.
# -----------------------------------------------------------------------------


def _venv_python(root: str) -> Optional[str]:
    """The repo venv interpreter (POSIX or Windows layout), or None."""
    base = Path(root) / ".venv"
    for cand in (base / "bin" / "python", base / "Scripts" / "python.exe",
                 base / "Scripts" / "python"):
        if cand.exists():
            return str(cand)
    return None


class _WorkerClient:
    """One persistent brain-worker subprocess, strict request/response."""

    def __init__(self, root: str, python_bin: str) -> None:
        self._root = root
        self._python = python_bin
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()
        self._seq = 0

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        import os as _os
        env = dict(_os.environ)
        env.pop("PYTHONPATH", None)  # venv is self-contained (see MCP launcher)
        env.setdefault("DUCKBOT_EMBEDDING", "lmstudio")
        env["PYTHONUNBUFFERED"] = "1"
        log_path = Path(self._root) / "data" / "brain-worker.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            err = open(log_path, "ab")
        except Exception:  # noqa: BLE001
            err = subprocess.DEVNULL
        self._proc = subprocess.Popen(
            [self._python, "-m", "src.brain_worker"],
            cwd=self._root, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=err, text=True, bufsize=1,
        )

    def call(self, op: str, args: Optional[dict] = None,
             timeout: Optional[float] = None) -> Any:
        """One round-trip. With a timeout, the call runs on a scratch
        thread so a wedged worker can't hang the agent loop."""
        if timeout is not None:
            box: dict = {}

            def _run() -> None:
                try:
                    box["result"] = self._call_blocking(op, args)
                except Exception as e:  # noqa: BLE001
                    box["error"] = e

            t = threading.Thread(target=_run, daemon=True)
            t.start()
            t.join(timeout)
            if t.is_alive():
                raise TimeoutError(f"brain worker {op} exceeded {timeout}s")
            if "error" in box:
                raise box["error"]
            return box.get("result")
        return self._call_blocking(op, args)

    def _call_blocking(self, op: str, args: Optional[dict] = None) -> Any:
        with self._lock:
            self.start()
            assert self._proc is not None
            assert self._proc.stdin is not None and self._proc.stdout is not None
            self._seq += 1
            req_id = self._seq
            self._proc.stdin.write(json.dumps(
                {"id": req_id, "op": op, "args": args or {}}) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            if not line:
                raise RuntimeError("brain worker exited unexpectedly")
            resp = json.loads(line)
            if resp.get("id") != req_id:
                raise RuntimeError("brain worker protocol desync")
            if not resp.get("ok"):
                raise RuntimeError(resp.get("error", "brain worker error"))
            return resp.get("result")

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass


class _BridgeBrain:
    """Brain-shaped facade over _WorkerClient.

    Only the surface the provider itself consumes (recall / remember /
    wake_up / stats); the brain tools go through the worker's dispatch
    op instead. Recall results come back as plain namespaces carrying
    the attributes prefetch reads (.text / .tier / .metadata / ...).
    """

    def __init__(self, client: _WorkerClient) -> None:
        self._client = client

    def recall(self, query: str, k: int = 5, **kwargs: Any) -> list:
        from types import SimpleNamespace
        rows = self._client.call("recall", {"query": query, "k": k, **kwargs})
        return [SimpleNamespace(**row) for row in rows or []]

    def remember(self, text: str, source_path: str = "<connector>",
                 force_tier: Any = None, **_: Any) -> Any:
        from types import SimpleNamespace
        tier = getattr(force_tier, "value", force_tier)
        out = self._client.call("remember", {
            "text": text, "source_path": source_path, "force_tier": tier})
        return SimpleNamespace(**(out or {}))

    def wake_up(self, **kwargs: Any) -> dict:
        return self._client.call("wake_up", kwargs) or {}

    def stats(self) -> Any:
        from types import SimpleNamespace
        return SimpleNamespace(**(self._client.call("stats") or {}))


# -----------------------------------------------------------------------------
# Plugin manifest
# -----------------------------------------------------------------------------


def register(ctx) -> None:
    """Register this memory provider with the Hermes plugin context.

    Pattern from plugins/memory/{honcho,holographic,bytetherover} — the
    plugin loader calls register(ctx) and the plugin pushes its
    provider into ctx.register_memory_provider(provider).
    """
    provider = DuckBotBrainProvider()
    logger.info(
        "[duckbot-brain] registering MemoryProvider (is_available=%s) — "
        "hooks: on_session_start, on_session_end, prefetch, sync_turn, "
        "system_prompt_block",
        provider.is_available(),
    )
    if hasattr(ctx, "register_memory_provider"):
        ctx.register_memory_provider(provider)
    else:
        # Flat-callable fallback (some loaders pass ctx as a callable).
        try:
            ctx(provider)
        except TypeError:
            # Fallback: store on a module-level handle.
            global _HANDOFF
            _HANDOFF = provider


_HANDOFF: Optional["DuckBotBrainProvider"] = None


# -----------------------------------------------------------------------------
# The provider
# -----------------------------------------------------------------------------


class DuckBotBrainProvider:
    """MemoryProvider that wraps the DuckBot Brain (Chroma + RRF + rerank + decay).

    Implements the Hermes MemoryProvider ABC. Each method is fast or
    background-safe so the agent loop is never blocked.
    """

    def __init__(self) -> None:
        self._brain: Optional[Brain] = None
        self._brain_ready: bool = False
        self._brain_lock = threading.Lock()
        # Bridge mode (v0.16.0): set by is_available() when this
        # interpreter can't host the Brain but the worker can.
        self._bridge_mode: bool = False
        self._worker: Optional[_WorkerClient] = None
        self._avail_probed: bool = False
        self._avail_result: bool = False
        self._unavailable_reason: str = ""
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="duckbot-brain-sync"
        )
        self._session_id: str = ""
        self._hermes_home: str = ""
        self._platform: str = ""
        self._agent_context: str = "primary"
        # v0.15.1: lifecycle event capture (best-effort; never blocks).
        self._events = None
        try:
            from pathlib import Path as _Path
            import os as _os
            from src.events import EventStore as _ES
            # __file__ is src/plugins/memory/duckbot_brain/__init__.py;
            # parents[4] gets us back to the repo root (where data/ lives).
            _events_path = _Path(
                _os.environ.get("DUCKBOT_EVENTS_DB")
                or (_Path(__file__).resolve().parents[4] / "data" / "events.db")
            )
            self._events = _ES(_events_path)
        except Exception:  # noqa: BLE001
            pass

    def _record_event(self, event_type: str, **kwargs) -> None:
        """Best-effort event write. Never raises — events are observability, not correctness."""
        if self._events is None or not self._session_id:
            return
        try:
            self._events.record_event(self._session_id, event_type, **kwargs)
        except Exception:  # noqa: BLE001
            pass

    # -- Identity ------------------------------------------------------------

    @property
    def name(self) -> str:
        return "duckbot-brain"

    # -- Lifecycle -----------------------------------------------------------

    def is_available(self) -> bool:
        """Is the Brain reachable from this process — directly or bridged?

        Direct mode: the Brain class + chromadb import in THIS
        interpreter (CLI sessions, the repo venv). Bridge mode: this
        interpreter can't host the Brain (the Hermes gateway's isolated
        bundled Python), so probe the brain worker under the repo venv
        Python instead — the provider stays live in the gateway too
        (v0.16.0; before this, the 2026-10-19 host gate made it
        self-disable there). The probe runs once per provider instance
        and the worker stays warm for the session.
        """
        if self._avail_probed:
            return self._avail_result
        self._avail_probed = True
        direct_err = ""
        try:
            from src.connectors.base import Brain as _  # noqa: F401
            # Touch src/ to confirm the repo path is set up correctly.
            import src
            assert hasattr(src, "__file__") and src.__file__
            import chromadb  # noqa: F401 — ChromaBackend hard dependency
            self._avail_result = True
            return True
        except Exception as e:  # noqa: BLE001
            direct_err = f"{type(e).__name__}: {e}"
        # Direct mode impossible in this interpreter — try the bridge.
        try:
            python_bin = _venv_python(_DUCKBOT_ROOT)
            if python_bin is None:
                raise RuntimeError(
                    f"no repo venv python under {_DUCKBOT_ROOT}")
            self._worker = _WorkerClient(_DUCKBOT_ROOT, python_bin)
            self._worker.call("ping", timeout=60)
            self._bridge_mode = True
            self._avail_result = True
            logger.info(
                "[duckbot-brain] bridge mode: brain worker live via %s",
                python_bin)
            return True
        except Exception as e:  # noqa: BLE001
            self._unavailable_reason = (
                f"direct import failed ({direct_err}); "
                f"bridge probe failed ({type(e).__name__}: {e})")
            logger.debug("DuckBot brain not available: %s",
                         self._unavailable_reason)
            return False

    def unavailable_reason(self) -> str:
        """Why is_available() is False — surfaced in Hermes' startup warning."""
        return (
            getattr(self, "_unavailable_reason", "")
            or "brain is not usable in this interpreter and the "
               "repo-venv bridge worker did not answer"
        )

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        """Connect + warm up.

        Per Hermes ABC: `hermes_home` and `platform` are guaranteed in kwargs.
        We may also receive `agent_context` ("primary"|"subagent"|"cron"|"flush")
        and skip writes for non-primary contexts.

        v0.15.2: kicks off brain warm-up in a background thread so the FIRST
        prefetch() doesn't block the agent's typing indicator waiting for
        ChromaDB + LM Studio + reranker to initialize. Previously the first
        prefetch called `_get_brain()` synchronously, which could take 2-5+
        seconds on cold start (file locks, model loads, embedding warmup) and
        was visible to the user as a stuck typing indicator.
        """
        self._session_id = session_id
        self._hermes_home = kwargs.get("hermes_home", "")
        self._platform = kwargs.get("platform", "cli")
        self._agent_context = kwargs.get("agent_context", "primary")

        # v0.15.1: log session start for lifecycle debugging.
        self._record_event(
            "session_start",
            context={
                "platform": self._platform,
                "agent_context": self._agent_context,
                "hermes_home": self._hermes_home or None,
            },
        )

        # Reset warm-up state so a re-initialize (e.g. new session_id)
        # triggers a fresh background warm-up.
        self._brain = None
        self._brain_ready = False

        # Kick off brain construction in the background. The agent loop
        # can call prefetch() immediately and get an empty result until
        # the brain is warm; subsequent prefetches are fast.
        try:
            self._executor.submit(self._warm_brain_background)
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to queue brain warm-up: %s", e)

        logger.info(
            "DuckBot brain provider initialized for session=%s platform=%s context=%s (warm-up queued)",
            session_id, self._platform, self._agent_context,
        )

    def _warm_brain_background(self) -> None:
        """Background brain construction. Runs on the executor thread.

        Sets `_brain_ready = True` on success. If construction raises, the
        error is logged but never propagated — prefetch will fall back to
        an empty result and the next prefetch (after brain is fixed) will
        succeed.
        """
        try:
            brain = self._get_brain()
            # Touch the brain with a no-op recall so we know embedding
            # providers + ChromaDB collections are initialized end-to-end.
            # Empty query is a no-op in the recall path, so this just
            # validates the pipeline works without returning junk.
            try:
                brain.recall("", k=1, rerank=False, decay=False)
            except Exception as warmup_err:  # noqa: BLE001
                logger.debug("brain warm-up probe failed (non-fatal): %s", warmup_err)
            self._brain_ready = True
            logger.info("DuckBot brain warm-up complete")
        except Exception as e:  # noqa: BLE001
            logger.warning("Brain warm-up failed (prefetch will return empty): %s", e)

    def shutdown(self) -> None:
        """Best-effort cleanup on agent exit."""
        if self._worker is not None:
            try:
                self._worker.close()
            except Exception:  # noqa: BLE001
                pass
        try:
            self._executor.shutdown(wait=True, cancel_futures=False)
        except Exception:
            pass

    # -- System prompt -------------------------------------------------------

    def system_prompt_block(self) -> str:
        """Static text injected into the system prompt.

        v0.14.0: delegated to the shared core so the OpenClaw adapter
        and the Hermes plugin advertise the same tool list.
        """
        return _tools.system_prompt_block()

    # -- Prefetch ------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return formatted context for the upcoming turn.

        Hermes ABC: should be fast (cached background result preferred).
        We do a tiny recall (k=3, no rerank, no decay) to stay under ~150ms.

        v0.15.2: if the brain is not yet warm (initial session-startup race),
        return "" immediately instead of blocking on brain construction. The
        agent's typing indicator stays active and the brain finishes warming
        in the background. Subsequent prefetches will hit the warm cache.
        """
        # Strip whitespace — a query of just spaces should be treated like
        # an empty query, otherwise we'd embed meaningless whitespace and
        # return random matches from the corpus.
        if not query or not query.strip():
            return ""
        query = query.strip()
        # Don't block the agent loop waiting for the brain to warm up.
        # If background warm-up hasn't completed yet, return empty.
        if not self._brain_ready or self._brain is None:
            logger.debug("prefetch: brain not ready yet, returning empty")
            return ""
        try:
            brain = self._get_brain()
            results = brain.recall(query, k=3, rerank=False, decay=False)
        except Exception as e:
            logger.debug("prefetch failed: %s", e)
            return ""
        if not results:
            return ""
        # Format for prompt injection. Keep short — Hermes truncates later.
        lines = ["[memory]"]
        for r in results:
            md = r.metadata or {}
            src = md.get("source_path", "?")
            tier = r.tier or "?"
            text = (r.text or "").strip().replace("\n", " ")
            if len(text) > 240:
                text = text[:237] + "..."
            lines.append(f"- ({tier}) {text}  [src: {src}]")
        return "\n".join(lines)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue a background recall for the NEXT turn.

        We don't yet use this — Hermes calls prefetch() synchronously each
        turn which is fast enough at our scale (LM Studio embeddings + 4k chunks).
        Override here to start a background query if recall becomes a bottleneck.
        """
        return

    # -- Sync (post-turn persistence) ---------------------------------------

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Persist a completed turn to the brain in the background.

        Per Hermes ABC: should be non-blocking. We submit to our executor.
        Per-agent-context: skip writes for non-primary contexts (cron, subagent,
        flush) so user representations don't get corrupted by system turns.
        """
        if self._agent_context != "primary":
            return
        if not user_content or not assistant_content:
            return
        # Background submit; do NOT await.
        try:
            self._executor.submit(self._sync_turn_blocking, user_content, assistant_content)
        except Exception as e:
            logger.warning("Failed to queue sync_turn: %s", e)

    def _sync_turn_blocking(self, user_content: str, assistant_content: str) -> None:
        """The actual sync logic, runs on the executor."""
        try:
            brain = self._get_brain()
            # Format as a memory entry. The Brain.remember() ingests
            # markdown with source path.
            entry = (
                f"# Turn @ {self._session_id}\n\n"
                f"**User:** {user_content}\n\n"
                f"**Assistant:** {assistant_content}\n"
            )
            source = f"hermes://{self._platform}/{self._session_id}"
            # Brain.remember() is SYNC — no asyncio.run wrapper needed.
            # asyncio.run on a non-coroutine raises "a coroutine was expected"
            # which silently dropped every sync_turn.
            brain.remember(text=entry, source_path=source)
        except Exception as e:
            logger.warning("sync_turn failed: %s", e)

    # -- Tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return OpenAI-function-call-style tool schemas.

        v0.14.0: delegates to the shared core surface so the Hermes
        plugin, the OpenClaw extension, and any future adapter all
        advertise the same 9 tools. The wrapper is purely the
        function-call-shape adaptation ({"type": "function", "function":
        {...}}) that Hermes expects.
        """
        return _tools.function_call_schemas()

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        """Dispatch a tool call from the agent. Returns JSON string.

        v0.14.0: delegates to the shared dispatch. Same surface as the
        OpenClaw adapter — agents get the same tool behavior whichever
        plugin loader they used.
        """
        # v0.15.1: pre-tool-use event capture (best-effort).
        self._record_event("pre_tool_use", tool_name=tool_name, args=args)
        # Per-tool rate limit (matches the MCP server + OpenClaw adapter).
        rl_err = _tools.check_rate_limit(tool_name)
        if rl_err is not None:
            return json.dumps(rl_err)
        # Bridge mode: the worker runs the shared dispatch against its
        # own Brain (this process can't). A test-provided real Brain in
        # _brain still wins, matching the override contract below.
        if self._bridge_mode and (
                self._brain is None or isinstance(self._brain, _BridgeBrain)):
            try:
                result = self._worker.call("dispatch", {
                    "name": tool_name, "args": args})
            except Exception as exc:  # noqa: BLE001
                self._record_event(
                    "tool_error", tool_name=tool_name, error=str(exc))
                raise
            try:
                self._record_event(
                    "post_tool_use", tool_name=tool_name, result=result)
            except Exception:  # noqa: BLE001
                pass
            return json.dumps(result, default=str)
        # Honor a per-provider brain override (test fixture path —
        # keeps `provider._brain = fake_brain` working in tests).
        try:
            if self._brain is not None:
                original = _tools._BRAIN
                _tools._BRAIN = self._brain
                try:
                    result = _tools.dispatch(tool_name, args)
                finally:
                    _tools._BRAIN = original
            else:
                result = _tools.dispatch(tool_name, args)
        except Exception as exc:
            # v0.15.1: surface tool errors so operators can grep
            # data/events.db for the failure trace.
            self._record_event("tool_error", tool_name=tool_name, error=str(exc))
            raise
        # v0.15.1: post-tool-use event capture (best-effort).
        # `result` is a dict; the EventStore truncates oversized values.
        try:
            self._record_event("post_tool_use", tool_name=tool_name, result=result)
        except Exception:  # noqa: BLE001
            pass
        return json.dumps(result, default=str)

    # -- Optional hooks (override to opt in) ---------------------------------

    def on_session_start(self) -> Optional[Dict[str, Any]]:
        """Hermes hook: end-of-session-startup. v0.14.0.

        Optional. If the plugin loader calls this at session start,
        returns the same shape as `brain_wake_up` so the agent gets
        context pre-loaded without an extra round-trip. Cheaper than
        the agent calling `brain_wake_up` itself because the wake-up
        runs in the plugin process (no MCP framing overhead).

        v0.15.3: added a soft wall-clock budget for this hook so a slow
        cold-start brain warm-up can't block the agent's typing
        indicator. If the brain warm-up exceeds `on_session_start_timeout_s`
        (default 6s), return a placeholder with `wake_up_truncated: True`
        and `status: 'background_warming'` so the agent continues with
        whatever context is available and the warm-up finishes in the
        background. The first prefetch() will then return real results
        on the next call.
        """
        import os as _os_oss
        import time as _time_oss
        timeout_s = float(_os_oss.environ.get(
            "DUCKBOT_ON_SESSION_START_TIMEOUT_S", "6.0"
        ))
        deadline = _time_oss.monotonic() + timeout_s

        # Brain may already be warm from a previous initialize() in this
        # session (e.g. if Hermes re-fires the hook). If so, return
        # immediately.
        if self._brain_ready and self._brain is not None:
            try:
                return self._brain.wake_up(
                    k=8,
                    include_blocks=True,
                    include_graph=True,
                    include_fsrs_review=True,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("on_session_start failed (brain warm): %s", e)
                return None

        # Cold path — wait briefly for warm-up to finish, but don't block
        # the agent loop past the timeout. Return a placeholder so the
        # agent can continue without context, and the first prefetch()
        # call will populate context on the next turn.
        wait_until = _time_oss.monotonic() + timeout_s
        while not self._brain_ready and _time_oss.monotonic() < wait_until:
            _time_oss.sleep(0.1)

        if not self._brain_ready or self._brain is None:
            logger.info(
                "on_session_start: brain not warm within %.1fs, returning placeholder",
                timeout_s,
            )
            return {
                "memories": [],
                "blocks": [],
                "graph_summary": {},
                "fsrs_review_queue": [],
                "stats": {},
                "wake_up_truncated": True,
                "wake_up_status": "background_warming",
                "wake_up_elapsed_ms": int(timeout_s * 1000),
            }

        try:
            return self._brain.wake_up(
                k=8,
                include_blocks=True,
                include_graph=True,
                include_fsrs_review=True,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("on_session_start failed: %s", e)
            return None

    def on_session_end(self, messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Hermes hook: end-of-session consolidation.

        v0.14.0: now actually does something. Walks the session
        messages for high-importance user statements (corrections,
        preferences, environment facts) and persists them as a single
        procedural-tier chunk via `brain_remember`. Returns a small
        summary dict the agent can surface to the user, or None if no
        actionable content was found.

        Per the Hermes ABC this runs synchronously, but `brain_remember`
        is non-blocking (queues on a daemon thread) so the agent loop
        doesn't stall.

        Per-agent-context: skip the write when the session isn't
        "primary" (cron / subagent / flush contexts) so we don't pollute
        the procedural tier with system-side chatter.
        """
        if not messages:
            return None
        if getattr(self, "_agent_context", "primary") != "primary":
            return {"skipped": "non-primary context"}

        # Pull durable-rule-shaped statements: user lines that mention
        # "always", "never", "must", "should", "I prefer", "I want", or
        # any line ≥ 60 chars ending in a period. Cheap regex — no LLM.
        import re
        durable_patterns = re.compile(
            r"\b(always|never|must|should|prefer|want|don'?t|do not)\b",
            re.IGNORECASE,
        )
        durable: list[str] = []
        for m in messages:
            role = (m.get("role") or m.get("sender") or "").lower()
            content = (m.get("content") or m.get("text") or "").strip()
            if role != "user" or not content:
                continue
            # Length window: 12..800 chars. The pattern (always/never/
            # must/should/prefer/want/don't) already filters most noise;
            # the length floor catches ultra-short fragments like "yes always"
            # which are usually mid-conversation acknowledgements rather than
            # durable rules. Previously 30 — too aggressive, dropped real
            # preferences like "I always use dark mode" (25 chars).
            if durable_patterns.search(content) and 12 <= len(content) <= 800:
                durable.append(content)

        if not durable:
            return None

        # Persist as one procedural chunk. Tag with session id so
        # future brain_recall can cite provenance.
        entry_lines = [f"# Session rules — {self._session_id}"]
        entry_lines.append("")
        for d in durable[:10]:  # cap to keep the chunk readable
            entry_lines.append(f"- {d}")
        entry = "\n".join(entry_lines)
        source = f"hermes://session-end/{self._session_id}"

        try:
            self._executor.submit(
                self._remember_blocking, entry, source, "procedural"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("on_session_end submit failed: %s", e)
            return None

        # v0.15.1: log session_end for lifecycle debugging. Includes
        # the count of durable rules persisted so an operator can
        # quickly verify the hook fired + worked.
        self._record_event(
            "session_end",
            context={
                "persisted": len(durable[:10]),
                "source": source,
                "tier": "procedural",
                "message_count": len(messages or []),
            },
        )

        return {
            "persisted": len(durable[:10]),
            "source": source,
            "tier": "procedural",
        }

    def _remember_blocking(self, text: str, source: str, tier: str) -> None:
        """Background remember() used by sync_turn and on_session_end."""
        try:
            brain = self._get_brain()
            from src.tier import Tier
            # Brain.remember() is SYNC — no asyncio.run wrapper needed.
            brain.remember(
                text=text,
                source_path=source,
                force_tier=Tier(tier),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("background remember failed: %s", e)

    # -- Internals -----------------------------------------------------------

    def _get_brain(self) -> Brain:
        """Lazy-construct the Brain singleton (or the bridge facade)."""
        if self._brain is None:
            if self._bridge_mode:
                assert self._worker is not None
                self._brain = _BridgeBrain(self._worker)  # type: ignore[assignment]
            else:
                self._brain = Brain()
        return self._brain

# -----------------------------------------------------------------------------
# Module-level hook shims (Hermes plugin loader calls these at module scope)
# Hermes plugin.yaml:  handler: on_session_start  → calls this function
# -----------------------------------------------------------------------------

_provider_instance = None


def _get_provider():
    """Return a cached DuckBotBrainProvider instance for module-level hooks."""
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = DuckBotBrainProvider()
    return _provider_instance


def on_session_start() -> dict:
    """Module-level shim for Hermes plugin loader.

    Instantiates the provider and delegates to the class method.
    Reads config from DUCKBOT_PLUGIN_CFG env var (JSON) or uses defaults.

    Returns the same dict as DuckBotBrainProvider.on_session_start().
    """
    import os, json as _json
    cfg = {}
    cfg_raw = os.environ.get("DUCKBOT_PLUGIN_CFG", "")
    if cfg_raw:
        try:
            cfg = _json.loads(cfg_raw)
        except Exception:
            pass
    # auto_wake_up defaults to True (matching the class method behavior)
    if not cfg.get("auto_wake_up", True):
        return {"status": "disabled"}
    try:
        p = _get_provider()
        p._session_id = os.environ.get("HERMES_SESSION_ID", "module-level")
        p._platform = os.environ.get("HERMES_PLATFORM", "cli")
        p._agent_context = os.environ.get("HERMES_AGENT_CONTEXT", "primary")
        return p.on_session_start() or {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}


def on_session_end(messages=None) -> dict:
    """Module-level shim for Hermes plugin loader.

    Delegates to DuckBotBrainProvider.on_session_end().
    If messages is None/empty, returns None (no-op, no error).
    """
    if not messages:
        return {"status": "skipped_empty_messages"}
    try:
        p = _get_provider()
        return p.on_session_end(messages) or {"status": "ok"}
    except Exception as e:
        return {"error": str(e)}


# Also expose at module top-level for direct import access
__version__ = "0.16.0"

__all__ = [
    "register",
    "DuckBotBrainProvider",
    "on_session_start",
    "on_session_end",
    "__version__",
]

