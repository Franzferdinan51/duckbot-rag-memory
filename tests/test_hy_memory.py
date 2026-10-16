"""
test_hy_memory.py — unit tests for the Hy-Memory writer adapter (Phase 0).

Covers the reply parser (documented JSON shape, bare list, code fences,
garbage) and the extractor's explicit-only / silent-failure contract with
a stubbed llm_client. No model is loaded by these tests.
"""
import src.hy_memory as hy
from src import llm_client


def test_parse_documented_shape():
    assert hy.parse_memories('{"memories": ["a fact", "a preference"]}') == [
        "a fact",
        "a preference",
    ]


def test_parse_bare_list_and_fences():
    assert hy.parse_memories('```json\n["one", "two"]\n```') == ["one", "two"]


def test_parse_dedupes_and_drops_empties():
    assert hy.parse_memories('{"memories": ["x", "x", " ", 42, "y"]}') == ["x", "y"]


def test_parse_garbage_returns_empty():
    assert hy.parse_memories("not json at all") == []
    assert hy.parse_memories("") == []
    assert hy.parse_memories(None) == []
    assert hy.parse_memories('{"wrong_key": []}') == []


def test_extract_requires_explicit_model():
    assert hy.extract_memories_hy("some chunk") == []
    assert hy.extract_memories_hy("some chunk", model="  ") == []
    assert hy.extract_memories_hy("   ", model="hy-memory") == []


def test_extract_roundtrip(monkeypatch):
    def fake(messages, **kwargs):
        assert kwargs["model"] == "hy-memory"
        assert kwargs["temperature"] == 0.0
        return '{"memories": ["Duckets grows strawberry kush"]}'

    monkeypatch.setattr(llm_client, "chat_completion", fake)
    out = hy.extract_memories_hy("we talked about plants", model="hy-memory")
    assert out == ["Duckets grows strawberry kush"]


def test_extract_failure_falls_back_to_empty(monkeypatch):
    monkeypatch.setattr(llm_client, "chat_completion", lambda *a, **k: None)
    assert hy.extract_memories_hy("chunk", model="hy-memory") == []
