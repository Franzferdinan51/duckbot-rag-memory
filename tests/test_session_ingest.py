"""Phase F: unified session ingest (grok + zcode normalizers, pipeline)."""
import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.memory import Memory
from src import session_ingest as si
from src import llm_client
from tests.test_reconcile import KeywordEmbedder


@pytest.fixture
def mem():
    tmp = Path(tempfile.mkdtemp(prefix="duckbot-ingest-test-"))
    return Memory(persist_dir=tmp / "chroma", embedder=KeywordEmbedder())


def _write_grok(path: Path):
    recs = [
        {"type": "system", "content": "You are Grok. Giant system prompt."},
        {"type": "user", "content": [{"type": "text", "text":
            "<user_info>\nOS: macos\n</user_info>\n<user_query>Duckets grows strawberry kush in the tent</user_query>"}]},
        {"type": "assistant", "content": "Noted: Duckets grows strawberry kush in the tent."},
    ]
    path.write_text("\n".join(json.dumps(r) for r in recs))


def _write_zcode(path: Path):
    rec = {"request": {"messages": [
        {"role": "system", "content": "huge system prompt"},
        {"role": "user", "content": "Duckets grows strawberry kush in the tent"},
    ]}, "response": {"text": "Noted."}}
    path.write_text(json.dumps(rec))


def test_normalizers_strip_boilerplate(tmp_path):
    g = tmp_path / "chat_history.jsonl"; _write_grok(g)
    t = si.grok_session_text(g)
    assert "Giant system prompt" not in t
    assert "user_info" not in t
    assert "Duckets grows strawberry kush" in t
    z = tmp_path / "rollout.jsonl"; _write_zcode(z)
    t2 = si.zcode_rollout_text(z)
    assert "huge system prompt" not in t2
    assert "ASSISTANT: Noted." in t2


def _dispatch(messages, **k):
    sys_prompt = messages[0]["content"]
    if "Extract explicit facts" in sys_prompt:
        return '{"memories": ["Duckets grows strawberry kush in the tent"]}'
    if "EXPERIENCE episodes" in sys_prompt:
        return '{"episodes": [{"task": "Duckets grows strawberry kush", "context": "", "steps": "grows kush in the tent", "outcome": "success"}]}'
    if "prospective intentions" in sys_prompt:
        return '{"intentions": []}'
    return '{"verdicts": []}'


async def test_ingest_pipeline_and_state(mem, tmp_path, monkeypatch):
    monkeypatch.setattr(llm_client, "chat_completion", _dispatch)
    g = tmp_path / "chat_history.jsonl"; _write_grok(g)
    state = tmp_path / "state.json"
    files = [("grok", g)]
    c1 = await si.ingest_files(mem, files, model="hy-memory-30b-a3b-mlx",
                               state_path=state)
    assert c1["files"] == 1 and c1["facts"] >= 1 and c1["episodes"] == 1
    res, _ = await mem.recall("Duckets grows strawberry kush", k=5)
    assert any("strawberry kush" in r.text for r in res)
    c2 = await si.ingest_files(mem, files, model="hy-memory-30b-a3b-mlx",
                               state_path=state)
    assert c2["skipped"] == 1 and c2["files"] == 0  # state-tracked


async def test_ingest_requires_model(mem, tmp_path):
    g = tmp_path / "chat_history.jsonl"; _write_grok(g)
    c = await si.ingest_files(mem, [("grok", g)], model=None)
    assert c["files"] == 0 and c["facts"] == 0
