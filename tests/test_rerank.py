"""
test_rerank.py — unit tests for the cross-encoder rerank pass (Layer 7).

Pattern from the existing tests/ folder (test_blocks.py, test_entities.py).
We exercise every code path without requiring an actual cross-encoder
model install — uses NoopBackend and a mock SentenceTransformersBackend.
"""

from __future__ import annotations

import pytest

from src.rerank import (
    DEFAULT_RERANK_MODEL,
    DEFAULT_LMSTUDIO_RERANK_MODEL,
    LMStudioBackend,
    NoopBackend,
    RerankResult,
    SentenceTransformersBackend,
    reset_backend,
    rerank,
    rerank_available,
    _parse_cohere_rerank_response,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_rerank_backend():
    """Ensure each test starts with a clean backend cache."""
    reset_backend()
    yield
    reset_backend()


@pytest.fixture
def sample_candidates() -> list[RerankResult]:
    return [
        RerankResult(
            id="c1",
            text="The weather in Huber Heights is sunny today.",
            tier="episodic",
            original_score=0.025,  # RRF rank 1
        ),
        RerankResult(
            id="c2",
            text="Unrelated chatter about a video game.",
            tier="episodic",
            original_score=0.016,  # RRF rank 2
        ),
        RerankResult(
            id="c3",
            text="Yesterday's discussion mentioned sunny weather too.",
            tier="episodic",
            original_score=0.010,  # RRF rank 3
        ),
    ]


# -----------------------------------------------------------------------------
# RerankResult
# -----------------------------------------------------------------------------


def test_rerank_result_to_dict():
    r = RerankResult(
        id="x",
        text="hello",
        tier="semantic",
        original_score=0.5,
        rerank_score=0.9,
        final_score=0.78,
    )
    d = r.to_dict()
    assert d["id"] == "x"
    assert d["tier"] == "semantic"
    assert d["original_score"] == 0.5
    assert d["rerank_score"] == 0.9
    assert d["final_score"] == 0.78


def test_rerank_result_defaults():
    r = RerankResult(id="y", text="x")
    assert r.tier == "unknown"
    assert r.original_score == 0.0
    assert r.rerank_score == 0.0
    assert r.final_score == 0.0
    assert r.metadata == {}


# -----------------------------------------------------------------------------
# NoopBackend
# -----------------------------------------------------------------------------


def test_noop_backend_preserves_order():
    be = NoopBackend()
    docs = ["first", "second", "third"]
    scores = be.score("anything", docs)
    assert len(scores) == 3
    # Noop gives descending scores so order is preserved when sorted by score.
    assert scores[0] > scores[1] > scores[2]


def test_noop_backend_handles_empty():
    be = NoopBackend()
    assert be.score("q", []) == []


def test_rerank_with_noop_keeps_input_order():
    candidates = [
        RerankResult(id="a", text="alpha", original_score=0.1),
        RerankResult(id="b", text="beta", original_score=0.05),
        RerankResult(id="c", text="gamma", original_score=0.02),
    ]
    out = rerank("q", candidates, backend=NoopBackend())
    # Noop backend can't actually rerank — we fall through to original scores.
    # Order should be preserved (a, b, c) because final_score = original_score.
    assert [r.id for r in out] == ["a", "b", "c"]


# -----------------------------------------------------------------------------
# Dict input normalization
# -----------------------------------------------------------------------------


def test_rerank_accepts_dict_input():
    candidates = [
        {"id": "d1", "text": "first doc", "tier": "episodic", "rrf_score": 0.05},
        {"id": "d2", "text": "second doc", "tier": "semantic", "rrf_score": 0.02},
    ]
    out = rerank("anything", candidates, backend=NoopBackend())
    assert all(isinstance(r, RerankResult) for r in out)
    assert out[0].id == "d1"
    assert out[1].id == "d2"


def test_rerank_accepts_chunk_id_key():
    """Some callers use 'chunk_id' instead of 'id'."""
    candidates = [
        {"chunk_id": "x1", "text": "doc one"},
        {"chunk_id": "x2", "text": "doc two"},
    ]
    out = rerank("q", candidates, backend=NoopBackend())
    assert out[0].id == "x1"
    assert out[1].id == "x2"


# -----------------------------------------------------------------------------
# Cross-encoder rerank (mocked)
# -----------------------------------------------------------------------------


class FakeCrossEncoderBackend:
    """Mimics SentenceTransformersBackend but returns deterministic scores."""

    def __init__(self, scores: list[float]):
        self._scores = scores
        self.name = "fake"

    def score(self, query: str, docs: list[str]) -> list[float]:
        # Cross-encoder relevance is query-dependent in real life; here
        # we just return the canned scores based on doc count.
        if len(self._scores) < len(docs):
            # Pad with zeros if caller passed more docs than we expected.
            return self._scores + [0.0] * (len(docs) - len(self._scores))
        return self._scores[: len(docs)]


def test_rerank_reorders_by_cross_encoder_score():
    """The cross-encoder (mocked) thinks doc 2 is most relevant.
    After rerank, doc 2 should be first even though it was RRF rank 2."""

    candidates = [
        RerankResult(id="d1", text="alpha", original_score=0.025, tier="episodic"),
        RerankResult(id="d2", text="beta", original_score=0.016, tier="episodic"),
        RerankResult(id="d3", text="gamma", original_score=0.010, tier="episodic"),
    ]
    # Cross-encoder: d3=0.9, d2=0.7, d1=0.1 (reverse order from RRF).
    fake = FakeCrossEncoderBackend([0.1, 0.7, 0.9])
    out = rerank("anything", candidates, backend=fake)

    # d3 should now be first because its rerank_score is highest.
    assert [r.id for r in out] == ["d3", "d2", "d1"]


def test_rerank_combines_original_and_rerank_scores():
    """final_score should blend both — not pure rerank, not pure RRF."""
    candidates = [
        RerankResult(id="d1", text="a", original_score=0.10),
        RerankResult(id="d2", text="b", original_score=0.05),
    ]
    fake = FakeCrossEncoderBackend([0.5, 0.9])
    out = rerank("q", candidates, backend=fake)

    # Both should have non-zero rerank_score AND non-zero final_score.
    for r in out:
        assert r.rerank_score > 0
        assert r.final_score > 0
    # Math: d1 final = 0.7*0.5 + 0.3*(0.10-min)/(0.10-0.05) = 0.65
    #       d2 final = 0.7*0.9 + 0.3*(0.05-min)/(0.10-0.05) = 0.63
    # d1 narrowly wins because its original_score advantage survives the blend.
    assert out[0].id == "d1"
    assert out[1].id == "d2"
    # And the rerank_score on each is preserved unchanged.
    by_id = {r.id: r.rerank_score for r in out}
    assert by_id["d1"] == 0.5
    assert by_id["d2"] == 0.9


def test_rerank_top_k_truncates():
    candidates = [
        RerankResult(id=f"d{i}", text=f"doc {i}", original_score=0.1 - i * 0.01)
        for i in range(10)
    ]
    fake = FakeCrossEncoderBackend([0.9 - i * 0.05 for i in range(10)])
    out = rerank("q", candidates, backend=fake, top_k=3)
    assert len(out) == 3
    # Top 3 by cross-encoder should be d0, d1, d2.
    assert [r.id for r in out] == ["d0", "d1", "d2"]


def test_rerank_handles_empty_input():
    assert rerank("q", [], backend=NoopBackend()) == []
    assert rerank("q", [], backend=FakeCrossEncoderBackend([])) == []


def test_rerank_failure_falls_back_to_input_order():
    """If the backend throws, the input order must be preserved (not raise)."""

    class BrokenBackend:
        name = "broken"

        def score(self, query, docs):
            raise RuntimeError("model crashed")

    candidates = [
        RerankResult(id="a", text="alpha", original_score=0.05),
        RerankResult(id="b", text="beta", original_score=0.03),
    ]
    out = rerank("q", candidates, backend=BrokenBackend())
    # We keep input order on failure — better than losing the query.
    assert [r.id for r in out] == ["a", "b"]


def test_rerank_backend_score_count_mismatch_falls_back():
    """If the backend returns the wrong number of scores, don't crash."""

    class MisalignedBackend:
        name = "misaligned"

        def score(self, query, docs):
            return [0.5, 0.9]  # only 2 scores for 3 docs

    candidates = [
        RerankResult(id="a", text="x", original_score=0.1),
        RerankResult(id="b", text="y", original_score=0.05),
        RerankResult(id="c", text="z", original_score=0.02),
    ]
    out = rerank("q", candidates, backend=MisalignedBackend())
    # Mismatch → preserve input order.
    assert [r.id for r in out] == ["a", "b", "c"]


# -----------------------------------------------------------------------------
# Backend resolution
# -----------------------------------------------------------------------------


def test_reset_backend_forces_re_resolution():
    # First call resolves and caches.
    be1 = _resolve_backend_safe()
    be2 = _resolve_backend_safe()
    assert be1 is be2
    # After reset, we get a (possibly different) instance.
    reset_backend()
    be3 = _resolve_backend_safe()
    # Same class — just a fresh instance.
    assert type(be3) is type(be1)


def _resolve_backend_safe():
    from src.rerank import _resolve_backend
    return _resolve_backend()


def test_rerank_available_with_noop():
    # If sentence-transformers isn't installed, fallback to noop → not "available".
    # (We can't guarantee the test env has or lacks the package, so we
    # check that the function returns a bool and that the noop path works.)
    result = rerank_available()
    assert isinstance(result, bool)


def test_default_rerank_model_is_qwen3_hf_repo():
    """Locked-in default for the Hugging Face / sentence-transformers path."""
    assert DEFAULT_RERANK_MODEL == "Qwen/Qwen3-Reranker-0.6B"


def test_default_lmstudio_rerank_model_is_local_alias():
    """Locked-in default for the LM Studio rerank path."""
    assert DEFAULT_LMSTUDIO_RERANK_MODEL == "qwen3-reranker-0.6b"


def test_resolve_backend_prefers_lmstudio_first(monkeypatch):
    """The local LM Studio reranker should win before sentence-transformers."""
    import src.rerank as rerank_module

    class FakeLMBackend:
        name = "lmstudio:fake"

        def score(self, query, docs):
            return [0.0] * len(docs)

    def fail_sentence_backend():
        raise AssertionError("sentence-transformers should not be selected first")

    monkeypatch.setattr(rerank_module, "LMStudioBackend", lambda: FakeLMBackend())
    monkeypatch.setattr(rerank_module, "SentenceTransformersBackend", fail_sentence_backend)
    rerank_module.reset_backend()

    backend = rerank_module._resolve_backend()
    assert backend.name == "lmstudio:fake"


def test_resolve_backend_does_not_cache_noop(monkeypatch):
    """If no backend is available yet, later availability should still be detected."""
    import src.rerank as rerank_module

    calls = {"lm": 0}

    class FakeNoop:
        name = "noop"

    class FakeLMBackend:
        name = "lmstudio:later"

        def score(self, query, docs):
            return [0.0] * len(docs)

    def fail_lm():
        calls["lm"] += 1
        raise RuntimeError("LM Studio not ready")

    monkeypatch.setattr(rerank_module, "LMStudioBackend", fail_lm)
    monkeypatch.setattr(rerank_module, "SentenceTransformersBackend", lambda: FakeNoop())
    rerank_module.reset_backend()

    first = rerank_module._resolve_backend()
    assert first.name == "noop"
    assert rerank_module._BACKEND is None
    assert calls["lm"] == 1

    monkeypatch.setattr(rerank_module, "LMStudioBackend", lambda: FakeLMBackend())
    second = rerank_module._resolve_backend()
    assert second.name == "lmstudio:later"
    assert rerank_module._BACKEND is second


# -----------------------------------------------------------------------------
# maybe_rerank — the integration hook used by src/query.py
# -----------------------------------------------------------------------------


class StubQueryResult:
    """Mimics src.query.QueryResult for maybe_rerank tests."""

    def __init__(self, cid, text, tier="episodic", rrf=0.0, metadata=None):
        self.chunk_id = cid
        self.text = text
        self.tier = tier
        self.rrf_score = rrf
        self.metadata = metadata or {}


def test_maybe_rerank_disabled_returns_input_unchanged(monkeypatch):
    """When explicitly disabled, no rerank happens, order is preserved."""
    monkeypatch.delenv("DUCKBOT_RERANK", raising=False)

    results = [
        StubQueryResult("a", "alpha", rrf=0.05),
        StubQueryResult("b", "beta", rrf=0.03),
    ]
    from src.rerank import maybe_rerank
    out = maybe_rerank("q", results, enabled=False)
    assert out == results


def test_maybe_rerank_enabled_reorders(monkeypatch):
    """When enabled, cross-encoder (noop) reranks by length-similarity heuristic.

    With NoopBackend, the noop preserves input order, but it should still
    return the same list of objects (not raise)."""
    monkeypatch.setenv("DUCKBOT_RERANK", "0")  # env says off
    results = [
        StubQueryResult("a", "alpha", rrf=0.05),
        StubQueryResult("b", "beta", rrf=0.03),
    ]
    from src.rerank import maybe_rerank
    # Explicit enabled=True overrides env var.
    out = maybe_rerank("q", results, enabled=True)
    # Should be a list of the same objects, reordered.
    assert len(out) == 2
    assert {r.chunk_id for r in out} == {"a", "b"}


def test_maybe_rerank_reads_env_var(monkeypatch):
    """When `enabled` is None, the env var decides."""
    monkeypatch.setenv("DUCKBOT_RERANK", "0")
    results = [StubQueryResult("a", "alpha")]
    from src.rerank import maybe_rerank
    out = maybe_rerank("q", results)  # enabled=None → reads env
    assert out == results  # env=0 → no rerank


# -----------------------------------------------------------------------------
# SentenceTransformersBackend import path (smoke test)
# -----------------------------------------------------------------------------


def test_sentence_transformers_backend_requires_package(monkeypatch):
    """If sentence-transformers isn't installed, the backend raises a helpful error."""
    # Hide the import by patching sys.modules.
    import sys
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    try:
        SentenceTransformersBackend()
    except (RuntimeError, ImportError, TypeError) as e:
        # Either RuntimeError (our message) or the import error itself.
        msg = str(e).lower()
        assert "sentence-transformers" in msg or "sentence_transformers" in msg
    else:
        pytest.fail("Expected an error when sentence-transformers is missing")


# -----------------------------------------------------------------------------
# LM Studio backend URL handling
# -----------------------------------------------------------------------------


def test_lmstudio_backend_uses_default_url(monkeypatch):
    """If LMSTUDIO_RERANK_URL is unset, falls back to the standard LM Studio port."""
    monkeypatch.delenv("LMSTUDIO_RERANK_URL", raising=False)
    monkeypatch.delenv("LMSTUDIO_RERANK_MODEL", raising=False)
    be = LMStudioBackend()
    assert be.url == "http://127.0.0.1:1234/v1/rerank"
    assert be.model == "qwen3-reranker-0.6b"


def test_lmstudio_backend_respects_env(monkeypatch):
    # delenv first to ensure clean slate regardless of .env or prior test state
    monkeypatch.delenv("LMSTUDIO_RERANK_URL", raising=False)
    monkeypatch.delenv("LMSTUDIO_RERANK_MODEL", raising=False)
    monkeypatch.setenv("LMSTUDIO_RERANK_URL", "http://localhost:9999/v1/rerank")
    monkeypatch.setenv("LMSTUDIO_RERANK_MODEL", "custom-reranker")
    be = LMStudioBackend()
    assert be.url == "http://localhost:9999/v1/rerank"
    assert be.model == "custom-reranker"


def test_lmstudio_backend_explicit_overrides_env(monkeypatch):
    # delenv first to ensure clean slate regardless of .env or prior test state
    monkeypatch.delenv("LMSTUDIO_RERANK_URL", raising=False)
    monkeypatch.delenv("LMSTUDIO_RERANK_MODEL", raising=False)
    monkeypatch.setenv("LMSTUDIO_RERANK_URL", "http://localhost:9999/v1/rerank")
    be = LMStudioBackend(url="http://other:8000/v1/rerank", model="x")
    assert be.url == "http://other:8000/v1/rerank"
    assert be.model == "x"


def test_lmstudio_backend_maps_cohere_response_by_index(monkeypatch):
    """_parse_cohere_rerank_response maps results by their `index` field, not
    by result order. If results come back sorted by relevance (doc1 > doc2 >
    doc0), scores must be placed at the correct doc index: [0.3, 0.9, 0.6]."""
    # Cohere returns results sorted by relevance, not in input order.
    # doc 0 = 0.3, doc 1 = 0.9, doc 2 = 0.6 (but results are [1, 2, 0]).
    response = {
        "results": [
            {"index": 1, "relevance_score": 0.9},
            {"index": 2, "relevance_score": 0.6},
            {"index": 0, "relevance_score": 0.3},
        ]
    }
    scores = _parse_cohere_rerank_response(response, n_docs=3)
    assert scores == [0.3, 0.9, 0.6], f"Expected [0.3, 0.9, 0.6], got {scores}"


def test_lmstudio_backend_handles_empty_results(monkeypatch):
    """Empty results array means no scores returned — treat all as 0.0."""
    scores = _parse_cohere_rerank_response({"results": []}, n_docs=3)
    assert scores == [0.0, 0.0, 0.0]


def test_lmstudio_backend_handles_partial_results(monkeypatch):
    """If server only returns top-N results, unscored docs get 0.0."""
    # Only 2 of 4 docs returned (top 2 by relevance).
    response = {
        "results": [
            {"index": 3, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.88},
        ]
    }
    scores = _parse_cohere_rerank_response(response, n_docs=4)
    assert scores == [0.88, 0.0, 0.0, 0.95], f"Expected [0.88, 0.0, 0.0, 0.95], got {scores}"


def test_lmstudio_backend_score_method(monkeypatch):
    """LMStudioBackend.score() returns scores for 3 docs."""
    from unittest.mock import patch

    be = LMStudioBackend(url="http://localhost:1234/v1/rerank", model="qwen3-reranker-0.6b")

    # Mock _score_async to return directly (bypasses real HTTP).
    async def fake_score_async(query, docs):
        return [0.1, 0.5, 0.9]

    with patch.object(be, "_score_async", fake_score_async):
        scores = be.score("query", ["a", "b", "c"])

    assert scores == [0.1, 0.5, 0.9]
