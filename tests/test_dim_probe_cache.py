"""Tests for the embedding-dim probe cache (v0.11.2).

When Memory() is constructed it has to learn the real dim of the active
embedding model — the defaults in src/embeddings.py are guesses (e.g.
1024 for lmstudio). The probe used to be `embed_one("dim probe")` on
every Memory() instantiation, which fires a real /v1/embeddings call
against LM Studio. In a long-lived process (Hermes, MCP server,
watcher, OpenClaw) Memory() is re-instantiated on every tool call, so
the probe spammed the LM Studio request log with "dim probe" entries
and triggered ERR_HTTP_HEADERS_SENT noise.

These tests confirm:
  1. The first Memory() init in a fresh process does call the embedder
     (one probe), and that result is cached.
  2. Subsequent Memory() inits in the same process do NOT call
     embed_one() at all when the (base_url, model) is already cached.
  3. The same is true for LMStudioEmbeddings._resolve_dim() at the
     embedder level.
  4. Different (base_url, model) tuples still probe independently.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.memory import Memory, _DIM_PROBE_CACHE
from src.embeddings import (
    LMStudioEmbeddings,
    _EMBEDDER_DIM_CACHE,
    reset_dim_cache,
)


class _CountingProvider:
    """Test embedder that counts embed_one() calls."""

    name = "lmstudio"
    base_url = "http://test-lmstudio:1234/v1"
    model = "test-embedding-model"

    def __init__(self, dim: int = 768):
        self.dim = dim
        self.embed_one_calls = 0

    async def embed_one(self, text: str) -> list[float]:
        self.embed_one_calls += 1
        return [0.1] * self.dim

    async def embed(self, texts):
        return [[0.1] * self.dim for _ in texts]


@pytest.fixture
def clean_cache():
    """Wipe the dim-probe caches before and after each test."""
    _DIM_PROBE_CACHE.clear()
    reset_dim_cache()  # also clears _EMBEDDER_DIM_CACHE
    yield
    _DIM_PROBE_CACHE.clear()
    reset_dim_cache()


@pytest.mark.asyncio
async def test_first_memory_init_probes(clean_cache, tmp_path):
    """The first Memory() init in a fresh process should probe once."""
    provider = _CountingProvider(dim=768)
    mem = Memory(persist_dir=tmp_path / "chroma", embedder=provider)

    await mem._ensure_initialized()

    assert provider.embed_one_calls == 1, (
        "first Memory() init should probe exactly once via embed_one('dim probe')"
    )
    assert provider.dim == 768
    # Both caches should now hold the resolved dim for this (base_url, model)
    assert _EMBEDDER_DIM_CACHE.get((provider.base_url, provider.model)) == 768
    assert _DIM_PROBE_CACHE.get((provider.base_url, provider.model)) == 768


@pytest.mark.asyncio
async def test_second_memory_init_skips_probe(clean_cache, tmp_path):
    """A second Memory() init in the same process must NOT re-probe."""
    provider = _CountingProvider(dim=768)

    mem1 = Memory(persist_dir=tmp_path / "chroma1", embedder=provider)
    await mem1._ensure_initialized()
    assert provider.embed_one_calls == 1

    # Second Memory(), different persist_dir, SAME provider instance.
    # (In real life the provider is freshly constructed per Memory(),
    # so the cache key is what protects us.)
    provider2 = _CountingProvider(dim=768)
    mem2 = Memory(persist_dir=tmp_path / "chroma2", embedder=provider2)
    await mem2._ensure_initialized()

    # The cache should short-circuit the probe for mem2 entirely.
    assert provider2.embed_one_calls == 0, (
        "cached (base_url, model) must skip the probe on the next Memory() init"
    )
    assert provider2.dim == 768


@pytest.mark.asyncio
async def test_embedder_resolve_dim_uses_cache(clean_cache):
    """LMStudioEmbeddings._resolve_dim() should consult the cache."""
    e = LMStudioEmbeddings(
        base_url="http://test-lmstudio:1234/v1",
        model="test-embedding-model",
        api_key="lm-studio",
    )
    # Pretend a previous probe already learned dim=512 for this endpoint.
    _EMBEDDER_DIM_CACHE[(e.base_url, e.model)] = 512

    await e._resolve_dim()

    assert e.dim == 512, (
        "cached dim should be applied without sending a probe request"
    )


@pytest.mark.asyncio
async def test_different_endpoints_probe_independently(clean_cache, tmp_path):
    """Two Memory() instances with different (base_url, model) should each probe."""
    provider_a = _CountingProvider(dim=384)
    provider_a.base_url = "http://endpoint-a:1234/v1"
    provider_a.model = "model-a"

    provider_b = _CountingProvider(dim=768)
    provider_b.base_url = "http://endpoint-b:1234/v1"
    provider_b.model = "model-b"

    mem_a = Memory(persist_dir=tmp_path / "a", embedder=provider_a)
    await mem_a._ensure_initialized()

    mem_b = Memory(persist_dir=tmp_path / "b", embedder=provider_b)
    await mem_b._ensure_initialized()

    assert provider_a.embed_one_calls == 1
    assert provider_b.embed_one_calls == 1
    # Cache holds both, keyed by identity
    assert _DIM_PROBE_CACHE[(provider_a.base_url, provider_a.model)] == 384
    assert _DIM_PROBE_CACHE[(provider_b.base_url, provider_b.model)] == 768


@pytest.mark.asyncio
async def test_probe_failure_is_cached(clean_cache, tmp_path):
    """A failed probe should be cached (sentinel) so we don't keep retrying."""

    class _FailingProvider(_CountingProvider):
        async def embed_one(self, text: str) -> list[float]:
            self.embed_one_calls += 1
            raise RuntimeError("simulated LM Studio outage")

    provider = _FailingProvider()
    mem = Memory(persist_dir=tmp_path / "chroma", embedder=provider)
    await mem._ensure_initialized()
    assert provider.embed_one_calls == 1

    # Second Memory() with same endpoint should NOT retry the probe.
    provider2 = _FailingProvider()
    mem2 = Memory(persist_dir=tmp_path / "chroma2", embedder=provider2)
    await mem2._ensure_initialized()
    assert provider2.embed_one_calls == 0, (
        "a failed probe must be cached so we don't keep retrying in the same process"
    )