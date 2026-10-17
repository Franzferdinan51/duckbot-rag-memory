"""Phase B: schema + intention layers and layer-aware recall."""
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.memory import Memory
from src import layers, llm_client
from tests.test_reconcile import KeywordEmbedder


@pytest.fixture
def mem():
    tmp = Path(tempfile.mkdtemp(prefix="duckbot-layers-test-"))
    m = Memory(persist_dir=tmp / "chroma", embedder=KeywordEmbedder())
    yield m


FACTS = [
    "Duckets grows strawberry kush in the tent",
    "Duckets grows Kazakhstan landrace in the tent",
    "Duckets tracks his grow with CannaAI",
]


async def test_schema_induction_stores_layer(mem, monkeypatch):
    for f in FACTS:
        await mem.remember(f, force_tier="semantic")
    monkeypatch.setattr(llm_client, "chat_completion",
                        lambda *a, **k: '{"schemas": ["Duckets grows cannabis at home and tracks the grow closely"]}')
    stored = await layers.induce_schema_layer(mem, model="hy-memory-30b-a3b-mlx")
    assert len(stored) == 1
    ctx = await layers.recall_context(mem, "Duckets grows")
    assert any("tracks the grow" in s["text"] for s in ctx["schemas"])


async def test_schema_induction_needs_min_facts(mem, monkeypatch):
    await mem.remember(FACTS[0], force_tier="semantic")
    monkeypatch.setattr(llm_client, "chat_completion",
                        lambda *a, **k: '{"schemas": ["should not happen"]}')
    assert await layers.induce_schema_layer(mem, model="m") == []


async def test_intention_extraction_and_recall(mem, monkeypatch):
    monkeypatch.setattr(
        llm_client, "chat_completion",
        lambda *a, **k: '{"intentions": [{"text": "Duckets will order more seeds", "due": "2026-11-01", "trigger": null}]}')
    out = await layers.extract_intention_layer(
        mem, ["note: order more seeds next month"], model="hy-memory-30b-a3b-mlx")
    assert out and out[0]["due"] == "2026-11-01"
    ctx = await layers.recall_context(mem, "Duckets order seeds")
    assert any(i["text"] == "Duckets will order more seeds" and i["due"] == "2026-11-01"
               for i in ctx["intentions"])


async def test_layers_fail_open_without_model(mem):
    assert await layers.induce_schema_layer(mem, model=None) == []
    assert await layers.extract_intention_layer(mem, ["x"], model=None) == []
