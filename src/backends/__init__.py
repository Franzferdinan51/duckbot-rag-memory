"""
backends/__init__.py — pluggable vector backend registry.

Public API:
  - get_backend(name=None, **kwargs) → VectorBackend instance
  - list_backends() → list[str]
  - register_backend(name, "pkg.mod.Class") → None  (runtime plugin)

Pattern source: MemPalace's `backends/base.py` (MIT).
https://github.com/MemPalace/mempalace/blob/develop/mempalace/backends/base.py

Selection is driven by DUCKBOT_BACKEND env var (default "chroma").
Available backends:
  - chroma   — local persistent ChromaDB (current default, MIT)
  - qdrant   — Qdrant (Apache-2.0) — stub, requires qdrant-client
  - lancedb  — LanceDB (Apache-2.0) — stub, requires lancedb
"""

from __future__ import annotations

import os
from typing import Any

from .base import (
    BackendStats,
    TierStats,
    VectorBackend,
    VectorHit,
    _EXTRA_REGISTRY,
    register_backend,
)


# Lazy registry of importable backends. We do NOT import them at module
# load time so that `pip install duckbot-rag-memory` doesn't force-install
# every backend's heavy native dependencies.
_REGISTRY: dict[str, str] = {
    "chroma": "src.backends.chroma.ChromaBackend",
    "qdrant": "src.backends.qdrant.QdrantBackend",
    "lancedb": "src.backends.lancedb.LanceDBBackend",
}


def list_backends() -> list[str]:
    """Return the names of all known backends (built-in + runtime-registered)."""
    return list({**_REGISTRY, **_EXTRA_REGISTRY}.keys())


def get_backend(name: str | None = None, **kwargs: Any) -> VectorBackend:
    """Resolve a backend by name. Default reads DUCKBOT_BACKEND env var.

    Args:
        name: Backend name (e.g. "chroma"). If None, reads env var.
        **kwargs: Forwarded to the backend constructor.

    Returns:
        A VectorBackend instance ready for use.

    Raises:
        ValueError: Unknown backend name.
        ImportError: Backend's native dependency is not installed.
    """
    name = name or os.environ.get("DUCKBOT_BACKEND", "chroma")
    known = {**_REGISTRY, **_EXTRA_REGISTRY}
    if name not in known:
        raise ValueError(
            f"unknown backend: {name!r}. Known: {sorted(known.keys())}"
        )
    module_path, class_name = known[name].rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(**kwargs)


__all__ = [
    "VectorBackend",
    "VectorHit",
    "BackendStats",
    "TierStats",
    "register_backend",
    "get_backend",
    "list_backends",
    "_REGISTRY",
    "_EXTRA_REGISTRY",
]
