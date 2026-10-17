"""Phase C: temporal-graph wiring — feed on write, augment on recall."""
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

_TMP = Path(tempfile.mkdtemp(prefix="duckbot-graph-test-"))
os.environ["DUCKBOT_GRAPH_PATH"] = str(_TMP / "graph.db")

from src.memory import Memory
from src import graphwire
from tests.test_reconcile import KeywordEmbedder


@pytest.fixture
def mem():
    tmp = Path(tempfile.mkdtemp(prefix="duckbot-graphmem-test-"))
    return Memory(persist_dir=tmp / "chroma", embedder=KeywordEmbedder())


def test_feed_and_neighbors():
    counts = graphwire.feed_text("Duckets lives in Huber Heights.", source="test")
    lines = graphwire.neighbors_for_query("where does Duckets live?")
    assert any("Duckets" in ln for ln in lines), f"counts={counts} lines={lines}"


def test_feed_safe_never_raises(monkeypatch):
    monkeypatch.setattr(graphwire, "feed_text", lambda *a, **k: 1 / 0)
    assert graphwire.feed_text_safe("anything") == {}


async def test_remember_graph_feed_opt_in(mem):
    await mem.remember("Duckets lives in Huber Heights.", force_tier="semantic",
                       graph_feed=True)
    lines = graphwire.neighbors_for_query("Duckets")
    assert any("Duckets" in ln for ln in lines)


async def test_recall_graph_augment(mem):
    await mem.remember("Duckets lives in Huber Heights.", force_tier="semantic",
                       graph_feed=True)
    results, _ = await mem.recall("Duckets", k=5, graph_augment=True)
    graph_hits = [r for r in results if (r.metadata or {}).get("via") == "graph"]
    assert graph_hits, "expected graph-augmented results"
    results2, _ = await mem.recall("Duckets", k=5)
    assert not [r for r in results2 if (r.metadata or {}).get("via") == "graph"]
