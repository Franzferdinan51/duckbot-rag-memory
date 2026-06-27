"""
Tests for the v0.11.2 LM Studio spam hotfix.

Covers:
  1. Embed-result cache (hit / miss / LRU eviction / disabled)
  2. Shared httpx.AsyncClient (singleton, reuse)
  3. Token-bucket rate limiter (sustained, burst, exhaustion, refill)
  4. Watcher content-hash dedup (skip on no-op change, re-ingest on real change)
  5. End-to-end embed() returns cached vector on second call with no network

All tests run without external network calls. We patch `httpx.AsyncClient.post`
to count requests and return canned responses.
"""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.embeddings import (
    _embed_cache,
    _rate_limiter,
    _TokenBucket,
    _EmbedCache,
    LMStudioEmbeddings,
    auto_detect_provider,
    close_http_client,
    get_embed_cache_stats,
    reset_embed_cache,
    reset_rate_limiter,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_globals():
    """Reset cache + limiter + http client between tests."""
    reset_embed_cache()
    reset_rate_limiter(rpm=10000)  # effectively unlimited for most tests
    yield
    reset_embed_cache()
    reset_rate_limiter()


def _canned_response(n_vectors=1, dim=4):
    """Build an httpx.Response-like dict for LM Studio's /v1/embeddings."""
    return {
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": [0.1 * (i + 1)] * dim, "index": i}
            for i in range(n_vectors)
        ],
        "model": "m",
        "usage": {"prompt_tokens": n_vectors, "total_tokens": n_vectors},
    }


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.headers = {}
        self.text = str(payload)

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


# ---------------------------------------------------------------------------
# 5. End-to-end: embed() returns cached vector on second call (no HTTP)
# ---------------------------------------------------------------------------

class TestEmbedEndToEndCaching:
    @pytest.mark.asyncio
    async def test_second_embed_uses_cache(self):
        """Two calls with same text — second should NOT make an HTTP request."""
        provider = LMStudioEmbeddings(
            base_url="http://127.0.0.1:9999/v1",
            model="test-model",
            api_key="dummy",
        )

        request_count = 0

        async def fake_post(self, url, **kwargs):
            nonlocal request_count
            request_count += 1
            n = len(kwargs.get("json", {}).get("input", []))
            return FakeResponse(_canned_response(n_vectors=n, dim=4))

        await close_http_client()

        with patch("httpx.AsyncClient.post", new=fake_post):
            v1 = await provider.embed(["hello world"])
            assert request_count == 1, f"Expected 1 request, got {request_count}"
            v2 = await provider.embed(["hello world"])
            assert request_count == 1, f"Expected 1 request after cache hit, got {request_count}"
            assert v1 == v2, "Cached vector must match first result"

    @pytest.mark.asyncio
    async def test_different_texts_each_trigger_request(self):
        provider = LMStudioEmbeddings(
            base_url="http://127.0.0.1:9999/v1",
            model="test-model",
            api_key="dummy",
        )

        request_count = 0

        async def fake_post(self, url, **kwargs):
            nonlocal request_count
            request_count += 1
            n = len(kwargs.get("json", {}).get("input", []))
            return FakeResponse(_canned_response(n_vectors=n, dim=4))

        await close_http_client()
        reset_embed_cache()

        with patch("httpx.AsyncClient.post", new=fake_post):
            await provider.embed(["hello", "world", "foo"])
            assert request_count == 1, f"3 texts in batch_size=32 → 1 request, got {request_count}"
            await provider.embed(["bar", "baz"])
            assert request_count == 2, f"2 new texts → +1 request, got {request_count}"
            await provider.embed(["hello", "world", "qux"])
            assert request_count == 3, f"1 new + 2 cached → +1 request, got {request_count}"

    @pytest.mark.asyncio
    async def test_cache_stats(self):
        reset_embed_cache()
        provider = LMStudioEmbeddings(
            base_url="http://127.0.0.1:9999/v1",
            model="test-model",
            api_key="dummy",
        )

        async def fake_post(self, url, **kwargs):
            n = len(kwargs.get("json", {}).get("input", []))
            return FakeResponse(_canned_response(n_vectors=n, dim=4))

        await close_http_client()

        with patch("httpx.AsyncClient.post", new=fake_post):
            await provider.embed(["text1", "text2", "text3"])
            stats = get_embed_cache_stats()
            assert stats["size"] == 3
            await provider.embed(["text1"])
            assert stats["size"] == 3


@pytest.mark.asyncio
async def test_auto_detect_provider_falls_back_when_lmstudio_probe_fails(monkeypatch):
    """A bad LM Studio embedding model should not poison provider selection."""
    import src.embeddings as emb

    class FakeLMStudioEmbeddings:
        def __init__(self, *args, **kwargs):
            self.base_url = kwargs.get("base_url", "http://127.0.0.1:1234/v1")
            self.model = kwargs.get("model", "bad-model")
            self.dim = 1024
            self.name = "lmstudio"

        async def _resolve_dim(self):
            return False

    monkeypatch.delenv("DUCKBOT_EMBEDDING", raising=False)
    monkeypatch.setenv("MINIMAX_API_KEY", "dummy-key")
    monkeypatch.setattr(emb, "LMStudioEmbeddings", FakeLMStudioEmbeddings)

    provider = await auto_detect_provider()

    assert provider.name == "minimax"


# ---------------------------------------------------------------------------
# 1. EmbedResult cache
# ---------------------------------------------------------------------------

class TestEmbedCache:
    def test_cache_miss_returns_none(self):
        c = _EmbedCache()
        assert c.get("hello", "model-a") is None

    def test_cache_hit_after_put(self):
        c = _EmbedCache()
        c.put("hello", "model-a", [1.0, 2.0])
        assert c.get("hello", "model-a") == [1.0, 2.0]

    def test_cache_distinguishes_models(self):
        c = _EmbedCache()
        c.put("hello", "model-a", [1.0])
        assert c.get("hello", "model-b") is None

    def test_cache_disabled_when_max_size_zero(self):
        c = _EmbedCache(max_size=0)
        c.put("hello", "model", [1.0])
        assert c.get("hello", "model") is None
        assert len(c) == 0

    def test_cache_lru_eviction(self):
        c = _EmbedCache(max_size=2)
        c.put("a", "m", [1.0])
        c.put("b", "m", [2.0])
        # Touch 'a' to mark as recently used
        assert c.get("a", "m") == [1.0]
        # Add 'c' — 'b' should be evicted (it's the LRU)
        c.put("c", "m", [3.0])
        assert c.get("b", "m") is None
        assert c.get("a", "m") == [1.0]
        assert c.get("c", "m") == [3.0]

    def test_cache_normalizes_text_by_hash(self):
        """Same content from different file objects should hit the cache."""
        c = _EmbedCache()
        c.put("hello world", "m", [1.0])
        # Simulate reading the same content via different paths
        assert c.get("hello world", "m") == [1.0]


# ---------------------------------------------------------------------------
# 2. Token bucket rate limiter
# ---------------------------------------------------------------------------

class TestTokenBucket:
    def test_constructs_without_running_loop(self):
        """_TokenBucket() should not require an active event loop."""
        b = _TokenBucket(rate_per_min=60, capacity=60)
        assert b._lock is None

    @pytest.mark.asyncio
    async def test_initial_burst_allows_capacity_tokens(self):
        b = _TokenBucket(rate_per_min=60, capacity=60)
        # All 60 should acquire instantly (no sleep)
        t0 = time.monotonic()
        for _ in range(60):
            await b.acquire()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.5, f"Burst took {elapsed}s, expected <0.5s"

    @pytest.mark.asyncio
    async def test_exhausted_bucket_sleeps(self):
        b = _TokenBucket(rate_per_min=60, capacity=2)
        await b.acquire()
        await b.acquire()
        # Third acquire should sleep ~1s (60 req/min = 1 req/s)
        t0 = time.monotonic()
        await b.acquire()
        elapsed = time.monotonic() - t0
        assert elapsed >= 0.8, f"Expected ~1s sleep, got {elapsed}s"

    @pytest.mark.asyncio
    async def test_refill_over_time(self):
        b = _TokenBucket(rate_per_min=600, capacity=2)
        await b.acquire()
        await b.acquire()
        # After 0.5s, should have refilled ~5 tokens (0.5s * 10/s)
        await asyncio.sleep(0.5)
        t0 = time.monotonic()
        for _ in range(5):
            await b.acquire()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.5, f"Refill didn't happen: {elapsed}s"


# ---------------------------------------------------------------------------
# 3. Shared httpx client (singleton)
# ---------------------------------------------------------------------------

class TestSharedClient:
    @pytest.mark.asyncio
    async def test_get_http_client_returns_singleton(self):
        from src.embeddings import _get_http_client
        c1 = await _get_http_client()
        c2 = await _get_http_client()
        assert c1 is c2

    @pytest.mark.asyncio
    async def test_close_then_reopen(self):
        from src.embeddings import _get_http_client
        c1 = await _get_http_client()
        await close_http_client()
        c2 = await _get_http_client()
        assert c1 is not c2

    @pytest.mark.asyncio
    async def test_reset_rate_limiter_clears_shared_http_client(self):
        import src.embeddings as embeddings
        from src.embeddings import _get_http_client, reset_rate_limiter

        await _get_http_client()
        assert embeddings._http_client is not None
        assert embeddings._http_client_lock is not None

        reset_rate_limiter()

        assert embeddings._http_client is None
        assert embeddings._http_client_lock is None


# ---------------------------------------------------------------------------
# 4. Watcher content-hash dedup
# ---------------------------------------------------------------------------

class TestWatcherContentHashDedup:
    @pytest.mark.asyncio
    async def test_no_op_save_skips_reingest(self, tmp_path: Path):
        """Touch a file with identical bytes — should be skipped, not re-embedded."""
        from src.watcher import sync_files

        md = tmp_path / "test.md"
        md.write_text("# Hello\n\nSome content here.", encoding="utf-8")

        # First sync — should add chunks
        state: dict = {"files": {}}
        with patch("src.watcher.Memory") as mock_mem_cls:
            mock_instance = mock_mem_cls.return_value
            mock_instance._ensure_initialized = AsyncMock(
                return_value=(MagicMockStore(), AsyncMock())
            )
            mock_instance.remember = AsyncMock(
                return_value=MagicMock(chunk_id="x")
            )
            stats = await sync_files([str(md)], state)
        assert stats["added"] >= 1, f"First sync should add chunks: {stats}"

        # Capture state after first sync
        first_state = state["files"][str(md)]
        assert "content_hash" in first_state

        # Touch the file (same content, but mtime will change).
        # Sleep 0.05s so mtime granularity doesn't accidentally match.
        await asyncio.sleep(0.05)
        original = md.read_text(encoding="utf-8")
        md.write_text(original, encoding="utf-8")

        # Second sync — should skip due to content_hash match.
        # Track embed calls to prove zero re-embedding happens.
        remember_calls: list = []
        async def remember_tracker(*args, **kwargs):
            remember_calls.append((args, kwargs))
            return MagicMock(chunk_id="y")
        mock_instance.remember = remember_tracker

        stats2 = await sync_files([str(md)], state)
        assert remember_calls == [], f"watcher re-embedded unchanged content: {len(remember_calls)} calls"
        assert stats2["added"] == 0 and stats2["updated"] == 0, \
            f"Content-hash dedup failed: {stats2}"
        # Confirm mtime was updated in state even though no re-ingest
        assert state["files"][str(md)]["mtime"] > first_state["mtime"]

    @pytest.mark.asyncio
    async def test_real_change_reingests(self, tmp_path: Path):
        from src.watcher import sync_files

        md = tmp_path / "test.md"
        md.write_text("# Hello\n\nOriginal content.", encoding="utf-8")

        state: dict = {"files": {}}
        with patch("src.watcher.Memory") as mock_mem_cls:
            mock_instance = mock_mem_cls.return_value
            mock_instance._ensure_initialized = AsyncMock(
                return_value=(MagicMockStore(), AsyncMock())
            )
            mock_instance.remember = AsyncMock(
                return_value=MagicMock(chunk_id="x")
            )
            await sync_files([str(md)], state)

        # Actually change content
        await asyncio.sleep(0.05)  # mtime granularity
        md.write_text("# Hello\n\nCompletely different content now.", encoding="utf-8")

        with patch("src.watcher.Memory") as mock_mem_cls:
            mock_instance = mock_mem_cls.return_value
            mock_instance._ensure_initialized = AsyncMock(
                return_value=(MagicMockStore(), AsyncMock())
            )
            mock_instance.remember = AsyncMock(
                return_value=MagicMock(chunk_id="y")
            )
            stats = await sync_files([str(md)], state)
        assert stats["updated"] + stats["added"] > 0


# ---------------------------------------------------------------------------
# Helpers for mocking
# ---------------------------------------------------------------------------

class MagicMock:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    def __getattr__(self, item):
        return MagicMock()


class MagicMockStore:
    def collection_for(self, tier):
        return MagicMock(delete=MagicMock())
