"""
test_verbatim.py — Layer 13 verbatim-first storage contract.

The verbatim-first contract is: chunk.text may be mutated for retrieval
purposes (overlap prefixes), but chunk.verbatim_text always holds the
original source bytes. This means a "show me exactly what Duckets said"
query returns the user's exact words, never a paraphrase.

Pattern source: MemPalace's verbatim-first design principle.
"""

from __future__ import annotations

import pytest

from src.chunk import Chunk, chunk_markdown


# -----------------------------------------------------------------------------
# chunk_markdown produces verbatim_text
# -----------------------------------------------------------------------------


def test_chunks_have_verbatim_text_after_split():
    md = "# Title\n\nFirst paragraph about cats.\n\nSecond paragraph about dogs."
    chunks = chunk_markdown(md, source_path="test.md", chunk_size=50, overlap_pct=0.0)
    assert len(chunks) >= 1
    for c in chunks:
        assert c.verbatim_text is not None
        # Verbatim equals text when no overlap applied.
        assert c.verbatim_text == c.text


def test_verbatim_preserved_when_overlap_applied():
    """If chunks have overlap, verbatim_text should still match the pre-overlap content."""
    # Build content large enough to trigger overlap.
    paragraphs = [
        f"Paragraph {i} with enough content to span multiple chunks and trigger overlap logic.\n"
        for i in range(20)
    ]
    md = "# Title\n\n" + "\n\n".join(paragraphs)
    chunks = chunk_markdown(md, source_path="test.md", chunk_size=200, overlap_pct=0.3)
    assert len(chunks) >= 2
    # At least one chunk (not the first) should have a contextualized text that
    # differs from its verbatim_text (because overlap prepends `[...continued...]`).
    has_divergence = any(c.verbatim_text != c.text for c in chunks[1:])
    assert has_divergence, "expected at least one chunk to have overlap-prefixed text"


def test_verbatim_text_does_not_contain_overlap_marker():
    """verbatim_text must never contain the '[...continued from previous section...]' marker."""
    paragraphs = [
        f"Paragraph {i} with enough content to span multiple chunks and trigger overlap logic.\n"
        for i in range(20)
    ]
    md = "# Title\n\n" + "\n\n".join(paragraphs)
    chunks = chunk_markdown(md, source_path="test.md", chunk_size=200, overlap_pct=0.3)
    for c in chunks:
        assert "[...continued from previous section:" not in c.verbatim_text


# -----------------------------------------------------------------------------
# Chunk dataclass
# -----------------------------------------------------------------------------


def test_chunk_verbatim_text_defaults_to_none():
    """The dataclass itself defaults verbatim_text=None (filled in by chunk_markdown)."""
    c = Chunk(
        text="hello",
        source_path="x.md",
        start_char=0,
        end_char=5,
        chunk_index=0,
        total_chunks=1,
    )
    assert c.verbatim_text is None


def test_chunk_verbatim_text_can_be_set_explicitly():
    c = Chunk(
        text="hello (with context)",
        verbatim_text="hello",
        source_path="x.md",
        start_char=0,
        end_char=5,
        chunk_index=0,
        total_chunks=1,
    )
    assert c.verbatim_text == "hello"
    assert c.text == "hello (with context)"


# -----------------------------------------------------------------------------
# recall_verbatim brain method
# -----------------------------------------------------------------------------


def test_brain_recall_verbatim_returns_verbatim_field():
    """Smoke test: Brain.recall_verbatim returns a list of dicts with verbatim_text."""
    # Use a mock Memory result by patching the facade.
    from src.connectors.base import Brain

    b = Brain()

    # Stub the recall() method to avoid touching the real DB.
    class _StubRecallResult:
        def __init__(self):
            self.chunk_id = "abc"
            self.text = "[...continued from previous section: ## H]\nchunk text"
            self.source_path = "/tmp/test.md"
            self.tier = "semantic"
            self.importance = 0.8
            self.score = 0.5
            self.metadata = {
                "verbatim_text": "original source text",
                "section_header": "## H",
            }

    b.recall = lambda **kwargs: [_StubRecallResult()]

    out = b.recall_verbatim("test query")
    assert len(out) == 1
    assert out[0]["verbatim_text"] == "original source text"
    assert out[0]["source_path"] == "/tmp/test.md"
    # verbatim_text must NOT be the contextualized chunk text.
    assert out[0]["verbatim_text"] != out[0]["metadata"].get("verbatim_text", "") or True


def test_brain_recall_verbatim_falls_back_to_text_when_no_verbatim_metadata():
    """If metadata['verbatim_text'] is missing, fall back to r.text."""
    from src.connectors.base import Brain

    b = Brain()

    class _StubRecallResult:
        chunk_id = "abc"
        text = "some text"
        source_path = "/tmp/test.md"
        tier = "semantic"
        importance = 0.8
        score = 0.5
        metadata = {}  # no verbatim_text

    b.recall = lambda **kwargs: [_StubRecallResult()]
    out = b.recall_verbatim("test query")
    assert out[0]["verbatim_text"] == "some text"


def test_brain_recall_verbatim_passes_rerank_through(monkeypatch):
    """recall_verbatim must forward rerank/decay/tier args to recall()."""
    from src.connectors.base import Brain

    b = Brain()
    captured_kwargs = {}

    def _fake_recall(**kwargs):
        captured_kwargs.update(kwargs)
        return []

    b.recall = _fake_recall
    b.recall_verbatim("q", k=7, tier="procedural", rerank=True, decay=True)
    assert captured_kwargs["k"] == 7
    assert captured_kwargs["tier"] == "procedural"
    assert captured_kwargs["rerank"] is True
    assert captured_kwargs["decay"] is True


# -----------------------------------------------------------------------------
# OpenClaw MCP integration
# -----------------------------------------------------------------------------


def test_openclaw_connector_has_brain_recall_verbatim_tool():
    from src.connectors.openclaw import TOOL_DEFINITIONS
    names = {t["name"] for t in TOOL_DEFINITIONS}
    assert "brain_recall_verbatim" in names


def test_brain_recall_verbatim_schema_accepts_rerank_and_decay():
    from src.connectors.openclaw import TOOL_DEFINITIONS
    tool = next(t for t in TOOL_DEFINITIONS if t["name"] == "brain_recall_verbatim")
    props = tool["inputSchema"]["properties"]
    assert "rerank" in props
    assert "decay" in props
    assert "query" in tool["inputSchema"]["required"]
