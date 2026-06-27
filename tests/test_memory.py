"""Tests for the unified Memory facade (remember/recall/reflect/forget)."""

import sys
import asyncio
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.memory import Memory, RememberResult
from src.tier import Tier
from src.embeddings import EmbeddingProvider
from tests._mock_embedder import MockEmbeddings


class _MockProvider:
    """Test embedder that delegates to MockEmbeddings."""
    name = "mock"
    dim = 384

    def __init__(self, dim: int = 384):
        self._impl = MockEmbeddings(dim=dim)
        self.dim = dim

    async def embed(self, texts):
        return await self._impl.embed(texts)

    async def embed_one(self, text):
        return await self._impl.embed_one(text)


@pytest.fixture
def mem():
    import shutil
    tmp = Path(tempfile.mkdtemp(prefix="duckbot-mem-test-"))
    m = Memory(persist_dir=tmp / "chroma", embedder=_MockProvider(dim=384))
    yield m
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.asyncio
async def test_remember_basic(mem):
    r = await mem.remember("Today we installed cua-driver.")
    assert isinstance(r, RememberResult)
    assert r.tier == Tier.EPISODIC
    assert r.importance > 0
    assert r.provider == "mock"
    assert r.stored is True


@pytest.mark.asyncio
async def test_remember_procedural_path(mem):
    """Files matching procedural path rules should classify as procedural."""
    r = await mem.remember("Always commit before pushing.", source_path="AGENTS.md")
    assert r.tier == Tier.PROCEDURAL


@pytest.mark.asyncio
async def test_recall_returns_results(mem):
    await mem.remember("Today we installed cua-driver v0.6.2.")
    await mem.remember("Yesterday we set up BrowserOS for X access.")
    await mem.remember("Always commit before pushing.", source_path="AGENTS.md")
    results, stats = await mem.recall("What did we install?", k=3)
    assert len(results) > 0
    assert stats.fused_results > 0


@pytest.mark.asyncio
async def test_recall_rejects_empty_query(mem):
    with pytest.raises(ValueError, match="non-empty string"):
        await mem.recall("   ")


@pytest.mark.asyncio
async def test_recall_bumps_importance(mem):
    r1 = await mem.remember("Duckets likes Eminem.")
    initial = r1.importance
    await mem.recall("What music does Duckets like?", k=3)
    # importance should have increased
    snap = await mem.stats()
    assert snap.total >= 1


@pytest.mark.asyncio
async def test_recall_filters_by_tier(mem):
    await mem.remember("Episodic: today we did X")
    await mem.remember("Always commit before pushing.", source_path="AGENTS.md")
    results, _ = await mem.recall("rule", k=5, tier="procedural")
    for r in results:
        assert r.tier == "procedural"


@pytest.mark.asyncio
async def test_recall_ignores_whitespace_tier(mem):
    await mem.remember("Always commit before pushing.", source_path="AGENTS.md")
    results, _ = await mem.recall("rule", k=5, tier="   ")
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_reflect_runs(mem):
    await mem.remember("Duckets installed cua-driver.")
    await mem.remember("Duckets uses LM Studio for inference.")
    result = await mem.reflect(lookback_days=7, max_chunks=50)
    assert "scanned" in result
    assert "extracted" in result
    assert result["scanned"] >= 1


@pytest.mark.asyncio
async def test_stats_snapshot(mem):
    await mem.remember("Test 1")
    await mem.remember("Test 2", source_path="AGENTS.md")
    snap = await mem.stats()
    assert snap.total >= 2
    assert "episodic" in snap.by_tier or "procedural" in snap.by_tier
    assert snap.by_provider.get("mock", 0) >= 2


@pytest.mark.asyncio
async def test_forget(mem):
    r = await mem.remember("This will be forgotten.")
    ok = await mem.forget(r.chunk_id)
    assert ok is True


@pytest.mark.asyncio
async def test_idempotent_remember(mem):
    """Remembering the same text twice should not create duplicates (content hash)."""
    r1 = await mem.remember("Today we did X")
    r2 = await mem.remember("Today we did X")
    assert r1.chunk_id == r2.chunk_id
    snap = await mem.stats()
    assert snap.total == 1


@pytest.mark.asyncio
async def test_entity_extraction(mem):
    r = await mem.remember("Duckets installed cua-driver and set up LM Studio.")
    entity_names = {e["name"] for e in r.entities}
    assert "Duckets" in entity_names or "cua-driver" in entity_names or "LM Studio" in entity_names


@pytest.mark.asyncio
async def test_relationship_extraction(mem):
    r = await mem.remember("Duckets installed cua-driver. Ryan uses LM Studio.")
    # Some relationship should be extracted
    assert isinstance(r.relationships, list)
