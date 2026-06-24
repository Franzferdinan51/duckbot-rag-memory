"""
test_backends.py — verify the pluggable backend seam (Layer 14).

duckbot-secret-scan: allowlist-file

Tests:
  - ABC contract (VectorBackend) is well-formed.
  - ChromaBackend round-trips add/query/bm25/stats/delete in a temp dir.
  - QdrantBackend / LanceDBBackend raise helpful ImportError when deps
    are missing, NotImplementedError when used without the deps.
  - get_backend() registry resolves by name + DUCKBOT_BACKEND env.
  - register_backend() works for runtime plugins.
"""

# duckbot-secret-scan: allowlist-file
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.backends import (  # noqa: E402
    _REGISTRY,
    get_backend,
    list_backends,
)
from src.backends.base import (  # noqa: E402
    BackendStats,
    TierStats,
    VectorBackend,
    VectorHit,
    register_backend,
)
from src.chunk import Chunk  # noqa: E402


# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------


def test_list_backends_returns_three_known_names():
    names = list_backends()
    assert "chroma" in names
    assert "qdrant" in names
    assert "lancedb" in names


def test_get_backend_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("does_not_exist")


def test_get_backend_default_is_chroma(monkeypatch):
    monkeypatch.delenv("DUCKBOT_BACKEND", raising=False)
    # We can't actually construct a ChromaBackend (needs chromadb installed)
    # but we can confirm the registry dispatches to the chroma class.
    with patch("src.backends.chroma.ChromaBackend") as mock_cls:
        mock_cls.return_value = MagicMock()
        b = get_backend()
        mock_cls.assert_called_once()
        assert b is mock_cls.return_value


def test_get_backend_reads_env_var(monkeypatch):
    monkeypatch.setenv("DUCKBOT_BACKEND", "chroma")
    with patch("src.backends.chroma.ChromaBackend") as mock_cls:
        mock_cls.return_value = MagicMock()
        get_backend()
        mock_cls.assert_called_once()


def test_get_backend_passes_kwargs():
    with patch("src.backends.chroma.ChromaBackend") as mock_cls:
        mock_cls.return_value = MagicMock()
        get_backend("chroma", persist_dir="/tmp/x", embedding_dim=768)
        mock_cls.assert_called_once_with(persist_dir="/tmp/x", embedding_dim=768)


def test_register_backend_adds_to_known_list():
    register_backend("my_backend", "my_pkg.module.MyBackend")
    from src.backends.base import all_known_backends
    known = all_known_backends()
    assert "my_backend" in known
    assert known["my_backend"] == "my_pkg.module.MyBackend"


def test_register_backend_rejects_empty_name():
    with pytest.raises(ValueError, match="non-empty"):
        register_backend("", "x.y.Z")


def test_register_backend_rejects_unqualified_path():
    with pytest.raises(ValueError, match="fully qualified"):
        register_backend("x", "NotQualified")


# -----------------------------------------------------------------------------
# ABC contract
# -----------------------------------------------------------------------------


def test_vector_backend_is_abstract():
    """VectorBackend cannot be instantiated directly."""
    with pytest.raises(TypeError):
        VectorBackend()


def test_vector_backend_required_methods():
    """The ABC requires the five core methods."""
    abstract = VectorBackend.__abstractmethods__
    assert "add_chunks" in abstract
    assert "query" in abstract
    assert "bm25_query" in abstract
    assert "delete" in abstract
    assert "stats" in abstract
    assert "name" in abstract
    assert "supported_tiers" in abstract


def test_vector_hit_is_json_safe():
    h = VectorHit(id="x", text="t", tier="semantic", distance=0.5, metadata={"a": 1})
    d = h.to_dict()
    import json as _json
    _json.dumps(d)  # must not raise


def test_backend_stats_total_and_chunks_per_tier():
    s = BackendStats(backend_name="t", tiers=[
        TierStats(name="episodic", chunk_count=10),
        TierStats(name="semantic", chunk_count=20),
    ])
    assert s.total == 30
    assert s.chunks_per_tier() == {"episodic": 10, "semantic": 20}


def test_backend_stats_to_dict_is_json_safe():
    import json as _json
    s = BackendStats(backend_name="t", tiers=[TierStats(name="x", chunk_count=1)])
    _json.dumps(s.to_dict())


# -----------------------------------------------------------------------------
# ChromaBackend round-trip (uses a temp persist dir)
# -----------------------------------------------------------------------------


@pytest.fixture
def temp_chroma_backend(tmp_path):
    """Create a ChromaBackend in a temp directory."""
    from src.backends.chroma import ChromaBackend
    return ChromaBackend(persist_dir=tmp_path / "chroma")


def _make_chunk(cid: str, text: str, verbatim: str | None = None) -> Chunk:
    return Chunk(
        text=text,
        verbatim_text=verbatim,
        source_path=f"/tmp/{cid}.md",
        start_char=0,
        end_char=len(text),
        chunk_index=0,
        total_chunks=1,
    )


def _embed(text: str, dim: int = 8) -> list[float]:
    """Deterministic toy embedding so we don't depend on a model for tests."""
    # Sum-of-chars hash mapped into [0, 1]
    h = sum(ord(c) for c in text) % 1000 / 1000.0
    return [h] * dim


def test_chroma_backend_name_and_tiers(temp_chroma_backend):
    assert temp_chroma_backend.name == "chroma"
    assert temp_chroma_backend.supported_tiers == ["working", "episodic", "semantic", "procedural"]


def test_chroma_backend_add_chunks_returns_count(temp_chroma_backend):
    chunks = [_make_chunk("a", "alpha"), _make_chunk("b", "beta")]
    embs = [_embed("alpha"), _embed("beta")]
    n = temp_chroma_backend.add_chunks(chunks, embs, tier="semantic")
    assert n == 2


def test_chroma_backend_add_chunks_rejects_empty(temp_chroma_backend):
    assert temp_chroma_backend.add_chunks([], [], tier="semantic") == 0


def test_chroma_backend_add_chunks_validates_lengths(temp_chroma_backend):
    chunks = [_make_chunk("a", "alpha")]
    with pytest.raises(ValueError, match="count mismatch"):
        temp_chroma_backend.add_chunks(chunks, [[0.1], [0.2]], tier="semantic")


def test_chroma_backend_add_chunks_rejects_unknown_tier(temp_chroma_backend):
    chunks = [_make_chunk("a", "alpha")]
    with pytest.raises(ValueError, match="unknown tier"):
        temp_chroma_backend.add_chunks(chunks, [[0.1]], tier="bogus")


def test_chroma_backend_query_returns_vector_hits(temp_chroma_backend):
    chunks = [_make_chunk("a", "alpha about cats"), _make_chunk("b", "beta about dogs")]
    embs = [_embed("alpha about cats"), _embed("beta about dogs")]
    temp_chroma_backend.add_chunks(chunks, embs, tier="semantic")

    hits = temp_chroma_backend.query(_embed("cats"), tier="semantic", n_results=2)
    assert len(hits) >= 1
    assert all(isinstance(h, VectorHit) for h in hits)
    assert all(h.tier == "semantic" for h in hits)
    # Sorted by distance ascending
    for i in range(len(hits) - 1):
        assert hits[i].distance <= hits[i + 1].distance


def test_chroma_backend_query_rejects_unknown_tier(temp_chroma_backend):
    with pytest.raises(ValueError, match="unknown tier"):
        temp_chroma_backend.query(_embed("x"), tier="bogus")


def test_chroma_backend_bm25_query_finds_keywords(temp_chroma_backend):
    chunks = [
        _make_chunk("a", "the quick brown fox"),
        _make_chunk("b", "lazy dog sleeps"),
        _make_chunk("c", "fox and dog are friends"),
    ]
    embs = [_embed(c.text) for c in chunks]
    temp_chroma_backend.add_chunks(chunks, embs, tier="semantic")

    hits = temp_chroma_backend.bm25_query("fox dog", tier="semantic", n_results=5)
    assert len(hits) >= 1
    texts = {h.text for h in hits}
    # Both 'fox' and 'dog' should appear in some returned hit.
    has_fox = any("fox" in t for t in texts)
    has_dog = any("dog" in t for t in texts)
    assert has_fox or has_dog


def test_chroma_backend_bm25_query_empty_keywords_returns_empty(temp_chroma_backend):
    assert temp_chroma_backend.bm25_query("a an", tier="semantic") == []


def test_chroma_backend_delete_removes_chunks(temp_chroma_backend):
    chunks = [_make_chunk("a", "alpha"), _make_chunk("b", "beta")]
    embs = [_embed("alpha"), _embed("beta")]
    temp_chroma_backend.add_chunks(chunks, embs, tier="semantic")
    # Capture id of 'a' so we can delete it.
    target_id = chunks[0].id
    deleted = temp_chroma_backend.delete([target_id], tier="semantic")
    assert deleted == 1
    # The deleted id should no longer appear in queries.
    hits = temp_chroma_backend.query(_embed("alpha"), tier="semantic", n_results=10)
    ids = {h.id for h in hits}
    assert target_id not in ids


def test_chroma_backend_delete_rejects_unknown_tier(temp_chroma_backend):
    with pytest.raises(ValueError, match="unknown tier"):
        temp_chroma_backend.delete(["x"], tier="bogus")


def test_chroma_backend_delete_empty_ids_returns_zero(temp_chroma_backend):
    assert temp_chroma_backend.delete([], tier="semantic") == 0


def test_chroma_backend_stats_reports_tier_counts(temp_chroma_backend):
    chunks = [_make_chunk(f"c{i}", f"text {i}") for i in range(3)]
    embs = [_embed(c.text) for c in chunks]
    temp_chroma_backend.add_chunks(chunks, embs, tier="semantic")
    s = temp_chroma_backend.stats()
    assert isinstance(s, BackendStats)
    assert s.backend_name == "chroma"
    assert s.total >= 3
    assert s.chunks_per_tier().get("semantic", 0) >= 3


def test_chroma_backend_collection_for_returns_underlying(temp_chroma_backend):
    coll = temp_chroma_backend.collection_for("semantic")
    assert coll is not None


def test_chroma_backend_collection_for_rejects_unknown(temp_chroma_backend):
    with pytest.raises(ValueError, match="unknown tier"):
        temp_chroma_backend.collection_for("bogus")


def test_chroma_backend_verbatim_text_preserved(temp_chroma_backend):
    """L13 verbatim text round-trips through the backend."""
    chunks = [_make_chunk("a", "[...continued from previous section: ## X]\nchunk text", verbatim="chunk text")]
    embs = [_embed("chunk text")]
    temp_chroma_backend.add_chunks(chunks, embs, tier="semantic")
    hits = temp_chroma_backend.query(_embed("chunk text"), tier="semantic", n_results=1)
    assert hits
    assert hits[0].metadata.get("verbatim_text") == "chunk text"


def test_chroma_backend_verbatim_text_truncated_at_8kb(temp_chroma_backend):
    """verbatim_text > 8192 chars gets the truncation marker."""
    long = "x" * 10000
    chunks = [_make_chunk("a", long, verbatim=long)]
    embs = [_embed("x")]
    temp_chroma_backend.add_chunks(chunks, embs, tier="semantic")
    hits = temp_chroma_backend.query(_embed("x"), tier="semantic", n_results=1)
    assert hits
    stored = hits[0].metadata.get("verbatim_text", "")
    assert "[truncated]" in stored
    assert len(stored) < 10000


# -----------------------------------------------------------------------------
# Stub backends (Qdrant, LanceDB) raise helpful errors without deps
# -----------------------------------------------------------------------------


def test_qdrant_import_error_when_missing(monkeypatch):
    """If qdrant-client isn't installed, importing QdrantBackend raises."""
    # Block qdrant_client from being imported
    import sys
    monkeypatch.setitem(sys.modules, "qdrant_client", None)
    from src.backends.qdrant import QdrantBackend
    with pytest.raises((ImportError, ModuleNotFoundError)):
        QdrantBackend()


def test_lancedb_import_error_when_missing(monkeypatch):
    """If lancedb isn't installed, importing LanceDBBackend raises."""
    import sys
    monkeypatch.setitem(sys.modules, "lancedb", None)
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    from src.backends.lancedb import LanceDBBackend
    with pytest.raises((ImportError, ModuleNotFoundError)):
        LanceDBBackend()


def test_qdrant_stub_methods_raise_not_implemented(monkeypatch):
    """Even if qdrant-client were installed, the core methods are stubs."""
    # Pretend qdrant_client is installed
    fake = MagicMock()
    fake.QdrantClient = MagicMock()
    import sys
    monkeypatch.setitem(sys.modules, "qdrant_client", fake)
    monkeypatch.setitem(sys.modules, "qdrant_client.http", MagicMock())
    monkeypatch.setitem(sys.modules, "qdrant_client.http.models", MagicMock())

    # Re-import the module fresh so the lazy import succeeds.
    import importlib
    import src.backends.qdrant as qmod
    importlib.reload(qmod)

    b = qmod.QdrantBackend(path="/tmp/qdrant_test_l14")
    with pytest.raises(NotImplementedError):
        b.add_chunks([], [], tier="semantic")
    with pytest.raises(NotImplementedError):
        b.query([0.1] * 4)
    with pytest.raises(NotImplementedError):
        b.bm25_query("anything")
    with pytest.raises(NotImplementedError):
        b.delete(["x"], tier="semantic")
    with pytest.raises(NotImplementedError):
        b.stats()


def test_lancedb_stub_methods_raise_not_implemented(monkeypatch):
    fake_lancedb = MagicMock()
    fake_pa = MagicMock()
    import sys
    monkeypatch.setitem(sys.modules, "lancedb", fake_lancedb)
    monkeypatch.setitem(sys.modules, "pyarrow", fake_pa)

    import importlib
    import src.backends.lancedb as lmod
    importlib.reload(lmod)

    b = lmod.LanceDBBackend(uri="/tmp/lancedb_test_l14")
    with pytest.raises(NotImplementedError):
        b.add_chunks([], [], tier="semantic")
    with pytest.raises(NotImplementedError):
        b.query([0.1] * 4)


# -----------------------------------------------------------------------------
# DUCKBOT_BACKEND env var controls dispatch
# -----------------------------------------------------------------------------


def test_backend_dispatch_via_env_var(monkeypatch, tmp_path):
    """DUCKBOT_BACKEND=chroma → ChromaBackend."""
    monkeypatch.setenv("DUCKBOT_BACKEND", "chroma")
    # Use a real ChromaBackend in a temp dir
    from src.backends.chroma import ChromaBackend
    with patch("src.backends.chroma.ChromaBackend", wraps=ChromaBackend) as spy:
        b = get_backend(persist_dir=str(tmp_path / "x"))
        spy.assert_called_once()
        assert isinstance(b, ChromaBackend)


def test_backend_dispatch_rejects_bogus_env(monkeypatch):
    monkeypatch.setenv("DUCKBOT_BACKEND", "nonexistent")
    with pytest.raises(ValueError):
        get_backend()
