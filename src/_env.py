"""Repo .env loading, shared by the CLI and the brain worker.

Historically only src/cli.py loaded the repo-root .env, which meant the
brain worker (the process Hermes actually talks to over MCP) silently ran
without any of the .env settings — DUCKBOT_EMBEDDING, DUCKBOT_GRAPH_FEED,
model pins. Both entry points now load it through here.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_repo_dotenv() -> None:
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
