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
import sys
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
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="duckbot-brain-sync"
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
        """Cheap check: is the Brain importable + reachable?

        Per the Hermes ABC, this should NOT make network calls (it gates
        activation in MemoryManager before the agent loop starts). We
        only verify the Brain class imports and the repo root is on
        the Python path — real readiness (LM Studio reachable, chroma
        initialized) is deferred to `initialize()` / first tool call,
        where failures surface as graceful `{"error": ...}` returns.
        """
        try:
            from src.connectors.base import Brain as _  # noqa: F401
            # Touch src/ to confirm the repo path is set up correctly.
            import src
            assert hasattr(src, "__file__") and src.__file__
            return True
        except Exception as e:
            logger.debug("DuckBot brain not available: %s", e)
            return False

    def initialize(self, session_id: str, **kwargs: Any) -> None:
        """Connect + warm up.

        Per Hermes ABC: `hermes_home` and `platform` are guaranteed in kwargs.
        We may also receive `agent_context` ("primary"|"subagent"|"cron"|"flush")
        and skip writes for non-primary contexts.
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

        # Lazy brain — only construct on first recall/sync to keep startup fast.
        self._brain = None
        logger.info(
            "DuckBot brain provider initialized for session=%s platform=%s context=%s",
            session_id, self._platform, self._agent_context,
        )

    def shutdown(self) -> None:
        """Best-effort cleanup on agent exit."""
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
        """
        # Strip whitespace — a query of just spaces should be treated like
        # an empty query, otherwise we'd embed meaningless whitespace and
        # return random matches from the corpus.
        if not query or not query.strip():
            return ""
        query = query.strip()
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
        """
        try:
            brain = self._get_brain()
            return brain.wake_up(
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
        """Lazy-construct the Brain singleton."""
        if self._brain is None:
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
__version__ = "0.15.1"

__all__ = [
    "register",
    "DuckBotBrainProvider",
    "on_session_start",
    "on_session_end",
    "__version__",
]

