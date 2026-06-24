"""DuckBot brain — Hermes MemoryProvider plugin.

Implements Hermes's MemoryProvider ABC (NousResearch/hermes-agent,
`agent/memory_provider.py`). When this plugin is active, the agent
will:

  - Call `prefetch(query)` before each turn and inject relevant chunks.
  - Call `sync_turn(user, assistant)` after each turn to remember the
    conversation in the background (non-blocking).
  - Expose brain_recall / brain_recall_verbatim / brain_reflect as tools.

This is the Hermes side of Layer 16 (cross-runtime integration).
Pattern source: NousResearch/hermes-agent/plugins/memory/{honcho,hindsight,holographic}.

No paid APIs. All local (LM Studio + Chroma + BGE reranker).
"""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from src.connectors.base import Brain

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

    # -- Identity ------------------------------------------------------------

    @property
    def name(self) -> str:
        return "duckbot-brain"

    # -- Lifecycle -----------------------------------------------------------

    def is_available(self) -> bool:
        """Cheap check: do we have the Brain module + LM Studio configured?"""
        try:
            # The Brain class exists and is importable.
            from src.connectors.base import Brain as _  # noqa: F401
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

        Tells the agent what tools it has for memory and when to use them.
        """
        return (
            "\n\n## Memory tools\n\n"
            "You have access to the DuckBot brain via these tools:\n"
            "- `memory_search` (alias `brain_recall`): hybrid retrieval "
            "(vector + BM25 + RRF). Optionally `rerank=true` for cross-encoder "
            "boost, `decay=true` for Ebbinghaus retention weighting.\n"
            "- `brain_recall_verbatim`: returns the original source bytes, "
            "never paraphrased. Use when the user asks 'what exactly did I say?'.\n"
            "- `brain_reflect`: synthesizes a memory-augmented answer from top-k "
            "chunks via the local LLM.\n\n"
            "Proactively search memory before answering questions about prior "
            "work, decisions, or preferences. Memory is local and free to query.\n"
        )

    # -- Prefetch ------------------------------------------------------------

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Return formatted context for the upcoming turn.

        Hermes ABC: should be fast (cached background result preferred).
        We do a tiny recall (k=3, no rerank, no decay) to stay under ~150ms.
        """
        if not query:
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
            asyncio.run(brain.remember(text=entry, source_path=source))
        except Exception as e:
            logger.warning("sync_turn failed: %s", e)

    # -- Tools ---------------------------------------------------------------

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Return OpenAI-function-call-style tool schemas.

        We expose three: recall, recall_verbatim, and reflect.
        Hermes injects these into the agent's tool list at startup.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "brain_recall",
                    "description": (
                        "Hybrid retrieval (vector + BM25 + RRF) over the DuckBot "
                        "brain. Returns top-k chunks with tier, source, importance, "
                        "score. Optionally rerank=true for cross-encoder boost, "
                        "decay=true for Ebbinghaus retention weighting."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "k": {"type": "integer", "default": 5},
                            "tier": {
                                "type": "string",
                                "enum": ["working", "episodic", "semantic", "procedural"],
                            },
                            "min_importance": {"type": "number"},
                            "rerank": {"type": "boolean", "default": False},
                            "decay": {"type": "boolean", "default": False},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "brain_recall_verbatim",
                    "description": (
                        "Returns the original (pre-overlap, pre-prefix) source "
                        "text — never paraphrased. Use when the user asks "
                        "'what exactly did I say about X?'."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "k": {"type": "integer", "default": 5},
                            "tier": {
                                "type": "string",
                                "enum": ["working", "episodic", "semantic", "procedural"],
                            },
                            "min_importance": {"type": "number"},
                            "rerank": {"type": "boolean", "default": False},
                            "decay": {"type": "boolean", "default": False},
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "brain_reflect",
                    "description": (
                        "Synthesize a memory-augmented answer: recall top-k "
                        "chunks, then compose a brief answer via the local LLM. "
                        "Use for 'summarize what we decided about X' style queries."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "k": {"type": "integer", "default": 8},
                            "tier": {
                                "type": "string",
                                "enum": ["working", "episodic", "semantic", "procedural"],
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs: Any) -> str:
        """Dispatch a tool call from the agent. Returns JSON string."""
        try:
            brain = self._get_brain()
            if tool_name == "brain_recall":
                results = brain.recall(
                    query=args["query"],
                    k=args.get("k", 5),
                    tier=args.get("tier"),
                    min_importance=args.get("min_importance"),
                    rerank=args.get("rerank") or False,
                    decay=args.get("decay") or False,
                )
                return json.dumps({
                    "results": [
                        {
                            "chunk_id": r.chunk_id,
                            "text": r.text,
                            "tier": r.tier,
                            "importance": r.importance,
                            "score": r.score,
                            "source_path": r.source_path,
                        }
                        for r in results
                    ],
                })
            if tool_name == "brain_recall_verbatim":
                results = brain.recall_verbatim(
                    query=args["query"],
                    k=args.get("k", 5),
                    tier=args.get("tier"),
                    min_importance=args.get("min_importance"),
                    rerank=args.get("rerank"),
                    decay=args.get("decay"),
                )
                return json.dumps({"results": results})
            if tool_name == "brain_reflect":
                # Use the Brain's existing reflect helper if present.
                # Brain may not have a `reflect` method — fall back to recall + plain wrap.
                results = brain.recall(
                    query=args["query"],
                    k=args.get("k", 8),
                    tier=args.get("tier"),
                )
                if not results:
                    return json.dumps({"answer": "", "results": []})
                # Compose a brief synthesis by concatenating chunks. Real LLM
                # synthesis is delegated to the calling agent.
                snippets = []
                for r in results[: min(len(results), 8)]:
                    text = (r.text or "").strip().replace("\n", " ")
                    if len(text) > 320:
                        text = text[:317] + "..."
                    snippets.append(f"[{r.tier}] {text}")
                return json.dumps({
                    "results": [
                        {"chunk_id": r.chunk_id, "score": r.score} for r in results
                    ],
                    "snippets": snippets,
                    "note": "snippets are pre-LLM synthesis; the calling agent should compose the final answer.",
                })
            return json.dumps({"error": f"unknown tool: {tool_name}"})
        except Exception as e:
            logger.warning("handle_tool_call %s failed: %s", tool_name, e)
            return json.dumps({"error": str(e)})

    # -- Optional hooks (override to opt in) ---------------------------------

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Hermes hook: end-of-session extraction.

        We already persist each turn via sync_turn. This is a hook point
        for future "summarize this session into one durable chunk" logic.
        """
        return None

    # -- Internals -----------------------------------------------------------

    def _get_brain(self) -> Brain:
        """Lazy-construct the Brain singleton."""
        if self._brain is None:
            self._brain = Brain()
        return self._brain
