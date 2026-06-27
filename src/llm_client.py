"""
llm_client.py — minimal chat-completion client for LM Studio.

This is an optional helper for consolidation / fact extraction in
src/consolidate.py. The main agent runtimes are OpenClaw and Hermes;
this client is only used when a host agent explicitly points it at an
already-loaded chat model. DuckBot itself does not launch a second one.

Design constraints:
- Uses the agent's own LM Studio server (same LMSTUDIO_URL + LMSTUDIO_KEY)
- Chat model is configured explicitly via DUCKBOT_CHAT_MODEL or passed
  into the helper by the caller; there is no built-in default model
- Zero new paid deps: only stdlib + httpx
- Sync interface: callers (consolidate) are sync; we run a tiny event loop
  on the existing httpx client
- Graceful fallback: any failure silently returns None so the caller
  falls back to the regex-only extraction path

NOT a general-purpose LLM client. For that, use a dedicated package.
This is just enough to do fact extraction in consolidate.py.
"""
from __future__ import annotations

import json
import os
from typing import Optional


def _resolve_url() -> str:
    """LM Studio base URL. Uses the same server as the embedding provider.
    Override with DUCKBOT_LLM_URL or LMSTUDIO_URL env var."""
    return (
        os.environ.get("DUCKBOT_LLM_URL")
        or os.environ.get("LMSTUDIO_URL")
        or "http://127.0.0.1:1234/v1"
    )


def _resolve_credential() -> str:
    """API key for LM Studio. Uses the same key as the embedding provider."""
    return (
        os.environ.get("DUCKBOT_LLM_KEY")
        or os.environ.get("LMSTUDIO_KEY")
        or os.environ.get("LMSTUDIO_API_KEY")
        or os.environ.get("LM_API_TOKEN")
        or ""
    )


def _resolve_model() -> str:
    """Chat model id for consolidation.

    There is no built-in default. The caller must either pass a model
    explicitly or set DUCKBOT_CHAT_MODEL to the agent's existing chat
    model. This keeps DuckBot from loading its own separate LLM.
    """
    return os.environ.get("DUCKBOT_CHAT_MODEL") or ""


def chat_completion(
    messages: list[dict],
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    timeout: float = 60.0,
) -> Optional[str]:
    """Run a chat completion against LM Studio. Returns the assistant
    message text, or None on any failure (so the caller can fall back
    to the regex-only path silently).

    Messages format: [{"role": "user"|"system"|"assistant", "content": "..."}, ...]
    """
    try:
        import httpx
    except ImportError:
        return None
    url = _resolve_url().rstrip("/") + "/chat/completions"
    credential = _resolve_credential()
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    resolved_model = (model or _resolve_model()).strip()
    if not resolved_model:
        return None
    payload["model"] = resolved_model
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if credential:
        headers["Authorization"] = f"Bearer {credential}"
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=payload, headers=headers)
        if r.status_code != 200:
            return None
        data = r.json()
    except Exception:
        return None
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None


def is_llm_available() -> bool:
    """Cheap health-check: is LM Studio reachable at the chat endpoint?
    Used to decide whether to attempt LLM extraction or fall back to
    regex-only. Does not verify the model is loaded — that check is
    deferred to the actual chat_completion call."""
    try:
        import httpx
    except ImportError:
        return False
    try:
        url = _resolve_url().rstrip("/") + "/models"
        credential = _resolve_credential()
        headers = {}
        if credential:
            headers["Authorization"] = f"Bearer {credential}"
        with httpx.Client(timeout=2.0) as client:
            r = client.get(url, headers=headers)
        return r.status_code == 200
    except Exception:
        return False
