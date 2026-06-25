"""
llm_client.py — minimal chat-completion client for LM Studio.

The brain embeds via LM Studio (src/embeddings.py) but until now
had no chat-completion client. This file adds one — used by the mem0
fact-extraction prompts in src/consolidate.py.

Design constraints:
- Local-first: defaults to LM Studio at http://127.0.0.1:1234/v1
- Zero new paid deps: only stdlib + httpx
- Reuses the shared httpx client in src/embeddings.py when available
- No API key required (LM Studio ignores auth headers)
- Sync interface: callers (consolidate) are sync; we run a tiny event loop
  on the existing httpx client

NOT a general-purpose LLM client. For that, use a dedicated package.
This is just enough to do fact extraction in consolidate.py.
"""
from __future__ import annotations

import json
import os
from typing import Optional


def _resolve_url() -> str:
    """LM Studio base URL. Defaults to localhost:1234/v1. Override with
    DUCKBOT_LLM_URL env var (also used by DUCKBOT_EMBEDDING=lmstudio)."""
    return (
        os.environ.get("DUCKBOT_LLM_URL")
        or os.environ.get("LMSTUDIO_URL")
        or "http://127.0.0.1:1234/v1"
    )


def _resolve_model() -> str:
    """Default model id. Override with DUCKBOT_LLM_MODEL or LMSTUDIO_MODEL."""
    return (
        os.environ.get("DUCKBOT_LLM_MODEL")
        or os.environ.get("LMSTUDIO_MODEL")
        or "qwen2.5-7b-instruct"
    )


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
    payload = {
        "model": model or _resolve_model(),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(url, json=payload)
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
    """Cheap health-check: is LM Studio reachable? Used to decide
    whether to attempt LLM extraction or fall back to regex-only."""
    try:
        import httpx
    except ImportError:
        return False
    try:
        with httpx.Client(timeout=2.0) as client:
            r = client.get(_resolve_url().rstrip("/") + "/models")
        return r.status_code == 200
    except Exception:
        return False
