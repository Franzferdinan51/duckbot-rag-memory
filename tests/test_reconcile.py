"""Phase A: Hy-Memory conflict reconciliation in the write path.

A changed fact (different text, same topic) must supersede the stale one
when a reconcile model judges it an update/contradiction — the near-dup
check (dist < 0.08) alone can never catch it. Without a reconcile model,
behavior must be exactly today's (both facts stay current). Fail-open:
a broken model reply keeps everything.
"""
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.memory import Memory
from src.hy_memory import parse_verdicts
from src import llm_client


VOCAB = [
    "duckets", "lives", "in", "huber", "heights", "dayton", "ohio",
    "grows", "strawberry", "kush", "tent", "the", "a", "his", "home",
]


class KeywordEmbedder:
    """Bag-of-words embedder: shared tokens -> small cosine distance."""
    name = "keyword"
    dim = 384

    def _vec(self, text):
        v = [0.0] * self.dim
        for tok in text.lower().replace(",", " ").replace(".", " ").split():
            if tok in VOCAB:
                v[VOCAB.index(tok)] += 1.0
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    async def embed(self, texts, timeout_ms=None):
        return [self._vec(t) for t in texts]

    async def embed_one(self, text, timeout_ms=None):
        return self._vec(text)


@pytest.fixture
def mem():
    tmp = Path(tempfile.mkdtemp(prefix="duckbot-reconcile-test-"))
    m = Memory(persist_dir=tmp / "chroma", embedder=KeywordEmbedder())
    yield m


OLD = "Duckets lives in Huber Heights Ohio"
NEW = "Duckets lives in Dayton Ohio"
OTHER = "Duckets grows strawberry kush in the tent"


def _stub_verdicts(monkeypatch, reply):
    monkeypatch.setattr(llm_client, "chat_completion", lambda *a, **k: reply)


async def _texts(m, q):
    res, _stats = await m.recall(q, k=10)
    return [r["text"] if isinstance(r, dict) else r.text for r in res]


async def test_updates_verdict_supersedes_stale_fact(mem, monkeypatch):
    await mem.remember(OLD, force_tier="semantic")
    _stub_verdicts(monkeypatch, '{"verdicts": [{"index": 0, "relation": "updates", "evidence": "moved"}]}')
    await mem.remember(NEW, force_tier="semantic", reconcile_model="hy-memory-30b-a3b-mlx")
    got = await _texts(mem, "Duckets lives in Ohio")
    assert NEW in got
    assert OLD not in got  # superseded -> filtered from recall


async def test_unrelated_verdict_keeps_both(mem, monkeypatch):
    await mem.remember(OLD, force_tier="semantic")
    _stub_verdicts(monkeypatch, '{"verdicts": [{"index": 0, "relation": "unrelated", "evidence": null}]}')
    await mem.remember(OTHER, force_tier="semantic", reconcile_model="hy-memory-30b-a3b-mlx")
    got = await _texts(mem, "Duckets")
    assert OLD in got and OTHER in got


async def test_no_model_preserves_legacy_behavior(mem, monkeypatch):
    await mem.remember(OLD, force_tier="semantic")
    _stub_verdicts(monkeypatch, '{"verdicts": [{"index": 0, "relation": "updates", "evidence": "moved"}]}')
    await mem.remember(NEW, force_tier="semantic")  # no reconcile_model
    got = await _texts(mem, "Duckets lives in Ohio")
    assert OLD in got and NEW in got  # model never consulted


async def test_model_failure_is_fail_open(mem, monkeypatch):
    await mem.remember(OLD, force_tier="semantic")
    _stub_verdicts(monkeypatch, None)  # transport failure shape
    await mem.remember(NEW, force_tier="semantic", reconcile_model="hy-memory-30b-a3b-mlx")
    got = await _texts(mem, "Duckets lives in Ohio")
    assert OLD in got and NEW in got


def test_parse_verdicts_strictness():
    assert parse_verdicts('{"verdicts": [{"index": 0, "relation": "same"}]}', 1)[0]["relation"] == "same"
    assert parse_verdicts('[{"index": 0, "relation": "contradicts"}]', 1)[0]["relation"] == "contradicts"
    assert parse_verdicts('{"verdicts": [{"index": 0, "relation": "bogus"}]}', 1) == []
    assert parse_verdicts('{"verdicts": [{"index": 3, "relation": "same"}]}', 1) == []
    assert parse_verdicts("not json", 1) == []
    assert parse_verdicts(None, 1) == []
    assert parse_verdicts('{"verdicts": [{"index": 0, "relation": "same"}, {"index": 0, "relation": "updates"}]}', 1).__len__() == 1
