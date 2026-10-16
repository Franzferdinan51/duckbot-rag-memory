"""
test_hy_memory_updates.py — Phase 5 add/update/retire ops: parser + contract.
"""
import src.hy_memory as hy
from src import llm_client


def test_parse_ops_documented_shape():
    out = hy.parse_memory_ops(
        '{"ops": [{"op": "add", "target": null, "text": "New fact"},'
        ' {"op": "retire", "target": "Old fact", "text": null}]}')
    assert out == [
        {"op": "add", "target": None, "text": "New fact"},
        {"op": "retire", "target": "Old fact", "text": None},
    ]


def test_parse_ops_drops_invalid():
    out = hy.parse_memory_ops(
        '{"ops": [{"op": "explode", "text": "x"},'
        ' {"op": "add", "text": "  "},'
        ' {"op": "update", "target": "old"},'
        ' {"op": "update", "target": "old", "text": "new"}]}')
    assert out == [{"op": "update", "target": "old", "text": "new"}]


def test_parse_ops_garbage():
    assert hy.parse_memory_ops("nope") == []
    assert hy.parse_memory_ops(None) == []


def test_plan_requires_model():
    assert hy.plan_memory_updates("chunk", ["existing"]) == []


def test_plan_roundtrip(monkeypatch):
    def fake(messages, **kwargs):
        assert kwargs["temperature"] == 0.0
        return ('{"ops": [{"op": "update", "target": "Duckets mines Pearl",'
                ' "text": "Duckets stopped mining on 2026-10-16"}]}')

    monkeypatch.setattr(llm_client, "chat_completion", fake)
    out = hy.plan_memory_updates(
        "mining is stopped now", ["Duckets mines Pearl"], model="hy-memory")
    assert out == [{"op": "update", "target": "Duckets mines Pearl",
                    "text": "Duckets stopped mining on 2026-10-16"}]
