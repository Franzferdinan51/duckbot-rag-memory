"""
llm_client.py — minimal chat-completion client for LM Studio.

This helper exists only for external jobs that already have a chat model
loaded and want to run explicit fact extraction. DuckBot itself does not
launch or default to a separate consolidation model.

Design constraints:
- Uses the caller's LM Studio server (same LMSTUDIO_URL + LMSTUDIO_KEY)
- Requires an explicit `model=` argument; there is no built-in default
- Zero new paid deps: only stdlib + httpx
- Sync interface: callers are sync; we run a tiny event loop on the
  existing httpx client
- Graceful fallback: any failure silently returns None so the caller can
  fall back to regex-only extraction or agent-supplied facts

NOT a general-purpose LLM client. For that, use a dedicated package.
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
    """Deprecated shim for callers that expect a resolver helper.

    DuckBot no longer reads a consolidation-model environment variable.
    Callers must pass `model=` explicitly if they want this helper to run.
    """
    return ""


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
