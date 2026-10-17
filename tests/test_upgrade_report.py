"""Phase G: upgrade acceptance report (census + graph stats)."""
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.memory import Memory
from src import upgrade_report as ur
from src import graphwire
from tests.test_reconcile import KeywordEmbedder


@pytest.fixture
def mem():
    tmp = Path(tempfile.mkdtemp(prefix="duckbot-report-test-"))
    return Memory(persist_dir=tmp / "chroma", embedder=KeywordEmbedder())


async def test_layer_census(mem):
    await mem.remember("Duckets lives in Huber Heights.",
                       force_tier="semantic")
    await mem.remember("Duckets moved to Dayton.",
                       force_tier="semantic")
    store, _ = await mem._ensure_initialized()
    coll = next(c for tier, c in store.all_collections.items()
                if (tier.value if hasattr(tier, "value") else str(tier)) == "semantic")
    got = coll.get(include=["metadatas"])
    ids, metas = got["ids"], got["metadatas"]
    assert ids
    coll.update(ids=[ids[0]], metadatas=[{
        **(metas[0] or {}), "superseded_by": "chunk-x",
        "superseded_via": "hy-reconcile"}])
    census = await ur.layer_census(mem)
    assert census["total"] >= 2
    assert census["superseded"] == 1
    assert census["superseded_via_reconcile"] == 1
    assert census["collections"]["semantic"] >= 2


def test_graph_stats_missing_db(tmp_path):
    stats = ur.graph_stats(tmp_path / "nope.db")
    assert stats["exists"] is False and stats["entities"] == 0


def test_graph_stats_after_feed(tmp_path, monkeypatch):
    db = tmp_path / "graph.db"
    monkeypatch.setenv("DUCKBOT_GRAPH_PATH", str(db))
    graphwire._GRAPH = None; graphwire._GRAPH_PATH = None
    graphwire.feed_text("Duckets lives in Huber Heights with Jessica.")
    stats = ur.graph_stats()
    assert stats["exists"] is True
    assert stats.get("entities", 0) >= 1
    graphwire._GRAPH = None; graphwire._GRAPH_PATH = None
