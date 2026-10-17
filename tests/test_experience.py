"""Phase D: experience episodes — record, recall, mine."""
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.memory import Memory
from src import experience, llm_client
from src.hy_memory import parse_episodes
from tests.test_reconcile import KeywordEmbedder


@pytest.fixture
def mem():
    tmp = Path(tempfile.mkdtemp(prefix="duckbot-exp-test-"))
    return Memory(persist_dir=tmp / "chroma", embedder=KeywordEmbedder())


async def test_record_and_recall_episode(mem):
    await experience.record_episode(
        mem, task="Duckets grows strawberry kush",
        steps="grows the kush in the tent and tracks it",
        outcome="success")
    eps = await experience.recall_experience(mem, "grows kush tent")
    assert eps and eps[0]["outcome"] == "success"
    assert "strawberry kush" in eps[0]["text"]


async def test_mine_episodes_with_stubbed_writer(mem, monkeypatch):
    monkeypatch.setattr(
        llm_client, "chat_completion",
        lambda *a, **k: '{"episodes": [{"task": "Duckets grows strawberry kush", "context": "home tent", "steps": "grows kush in the tent", "outcome": "success"}]}')
    stored = await experience.mine_episodes(
        mem, "session log text", model="hy-memory-30b-a3b-mlx")
    assert len(stored) == 1
    eps = await experience.recall_experience(mem, "grows kush")
    assert len(eps) == 1


def test_parse_episodes_strictness():
    assert parse_episodes('{"episodes": [{"task": "t", "steps": "s"}]}')[0]["outcome"] == "partial"
    assert parse_episodes('{"episodes": [{"task": "", "steps": "s"}]}') == []
    assert parse_episodes('{"episodes": [{"task": "t"}]}') == []
    assert parse_episodes("junk") == []


async def test_mine_fail_open_without_model(mem):
    assert await experience.mine_episodes(mem, "text", model=None) == []
