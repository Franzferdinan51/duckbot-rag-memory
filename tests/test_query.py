import sys
import asyncio
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.store import MemoryStore
from tests._mock_embedder import MockEmbeddings as LocalEmbeddings
from src.query import hybrid_query, _rrf_score
from src.chunk import chunk_markdown
from src.tier import Tier
import shutil
import tempfile


# Use a temp dir for tests so we don't clobber real ChromaDB


@pytest.fixture
def store():
    test_dir = Path(tempfile.mkdtemp(prefix="duckbot-rag-test-"))
    s = MemoryStore(persist_dir=test_dir / "chroma", embedding_dim=384, embedding_provider_name="local-test")
    s.reset()
    yield s
    # Cleanup
    import shutil
    shutil.rmtree(test_dir, ignore_errors=True)


@pytest.fixture
def embedder():
    return LocalEmbeddings(dim=384) if False else None


def make_dummy_chunks():
    """Create synthetic chunks that match the same content across tiers."""
    docs = {
        "episodic.md": [
            "On 2026-06-22 we installed cua-driver version 0.6.2 from trycua/cua.",
            "Duckets said 'use cloud-only' on 2026-06-10.",
        ],
        "procedural.md": [
            "Always commit before pushing to origin. Never skip tests.",
            "When kill_app is called with pid 1, refuse it.",
        ],
        "semantic.md": [
            "Duckets' home address: 123 Maple Street, Springfield, IL 62701.",
            "RTX Spark costs $4,499 for inference workloads.",
        ],
    }
    chunks = []
    for path, texts in docs.items():
        for text in texts:
            from src.chunk import Chunk
            chunks.append(Chunk(
                text=text,
                source_path=f"/fake/{path}",
                start_char=0,
                end_char=len(text),
                chunk_index=0,
                total_chunks=1,
                section_header=None,
            ))
    return chunks


def test_rrf_score_basic():
    assert _rrf_score(None) == 0.0
    assert abs(_rrf_score(1) - 1/61) < 1e-6
    assert abs(_rrf_score(2) - 1/62) < 1e-6


def test_rrf_score_higher_for_higher_rank():
    assert _rrf_score(1) > _rrf_score(5)
    assert _rrf_score(5) > _rrf_score(50)


@pytest.mark.asyncio
async def test_hybrid_query_returns_results(store):
    """End-to-end: embed chunks, query, verify retrieval."""
    chunks = make_dummy_chunks()
    embedder = LocalEmbeddings(dim=384)
    # Embed and add to appropriate tiers
    for tier, tier_chunks in [
        (Tier.EPISODIC, [c for c in chunks if "episodic" in c.source_path]),
        (Tier.PROCEDURAL, [c for c in chunks if "procedural" in c.source_path]),
        (Tier.SEMANTIC, [c for c in chunks if "semantic" in c.source_path]),
    ]:
        if not tier_chunks:
            continue
        embs = await embedder.embed([c.text for c in tier_chunks])
        await store.add_chunks(tier_chunks, embs, tier)

    results, stats = await hybrid_query(
        "Where does Duckets live?",
        store=store, embedder=embedder, n_results=3,
    )
    assert len(results) > 0
    # The semantic address chunk should rank highly
    top_texts = " ".join(r.text for r in results[:3])
    assert "123 Maple" in top_texts or "Springfield" in top_texts


@pytest.mark.asyncio
async def test_hybrid_query_bm25_catches_keyword(store):
    """BM25 fallback should help when the query has unique keywords."""
    chunks = make_dummy_chunks()
    embedder = LocalEmbeddings(dim=384)
    for tier, tier_chunks in [
        (Tier.EPISODIC, [c for c in chunks if "episodic" in c.source_path]),
        (Tier.PROCEDURAL, [c for c in chunks if "procedural" in c.source_path]),
        (Tier.SEMANTIC, [c for c in chunks if "semantic" in c.source_path]),
    ]:
        if not tier_chunks:
            continue
        embs = await embedder.embed([c.text for c in tier_chunks])
        await store.add_chunks(tier_chunks, embs, tier)

    # "cua-driver" is a unique keyword — BM25 should find the episodic chunk
    results, _ = await hybrid_query(
        "cua-driver version",
        store=store, embedder=embedder, n_results=5,
    )
    top_texts = " ".join(r.text for r in results[:3])
    assert "cua-driver" in top_texts


def test_store_stats_starts_zero(store):
    stats = store.stats()
    assert stats.total == 0
    assert stats.working == 0
    assert stats.episodic == 0
    assert stats.semantic == 0
    assert stats.procedural == 0


@pytest.mark.asyncio
async def test_store_add_and_count(store):
    from src.chunk import Chunk
    chunks = [
        Chunk(text=f"test chunk {i}", source_path=f"test{i}.md",
              start_char=0, end_char=10, chunk_index=i, total_chunks=3)
        for i in range(3)
    ]
    embs = [[0.1] * 384 for _ in chunks]  # LocalEmbeddings dim
    await store.add_chunks(chunks, embs, Tier.SEMANTIC)
    stats = store.stats()
    assert stats.semantic == 3
    assert stats.total == 3


def teardown_module():
    pass  # per-test cleanup is in the store fixture