import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunk import chunk_markdown, _recursive_split, _find_section_header, Chunk


SAMPLE_MD = """# Top Title

Some intro paragraph.

## Section One

First paragraph in section one.
Has multiple lines but should stay together.

### Subsection

- list item 1
- list item 2

```python
def hello():
    return "world"
```

## Section Two

Last section content.
"""


def test_chunks_respect_headers():
    chunks = chunk_markdown(SAMPLE_MD, source_path="test.md", chunk_size=64)
    # No chunk should be empty
    for c in chunks:
        assert c.text.strip(), f"empty chunk: {c!r}"
    # At least one chunk should carry the "## Section One" header
    section_one_chunk = next(
        (c for c in chunks if "Section One" in (c.section_header or "")),
        None,
    )
    assert section_one_chunk is not None, "should have a chunk with 'Section One' header"


def test_chunks_have_stable_ids():
    chunks1 = chunk_markdown(SAMPLE_MD, source_path="test.md")
    chunks2 = chunk_markdown(SAMPLE_MD, source_path="test.md")
    ids1 = [c.id for c in chunks1]
    ids2 = [c.id for c in chunks2]
    assert ids1 == ids2, "chunk IDs must be stable across runs"


def test_chunk_size_respected():
    chunks = chunk_markdown(SAMPLE_MD, source_path="test.md", chunk_size=128)
    # With 128 tokens target (~450 chars), no single chunk should be wildly oversized
    for c in chunks:
        # Allow 2x target for last chunk of section
        assert c.char_count <= 128 * 3.5 * 2, f"chunk too large: {c.char_count} chars"


def test_recursive_split_handles_short_text():
    chunks = _recursive_split("hello world", [". ", " "], chunk_size=100)
    assert chunks == ["hello world"]


def test_find_section_header():
    text = "# H1\n\n## H2\n\ncontent here"
    offset = text.find("content")
    header = _find_section_header(text, offset)
    assert "## H2" in header


def test_no_orphan_headers():
    chunks = chunk_markdown(SAMPLE_MD, source_path="test.md", chunk_size=512)
    for c in chunks:
        # If a chunk's first non-empty line is a header, the chunk should
        # also include the content right after it
        if c.section_header and c.text.lstrip().startswith("#"):
            # Header is in section_header metadata
            assert c.text.strip().startswith(c.section_header.split("\n")[0])


def test_total_chunks_filled():
    chunks = chunk_markdown(SAMPLE_MD, source_path="test.md")
    for c in chunks:
        assert c.total_chunks == len(chunks)


def test_chunk_index_sequential():
    chunks = chunk_markdown(SAMPLE_MD, source_path="test.md")
    for i, c in enumerate(chunks):
        assert c.chunk_index == i


def test_overlap_adds_context():
    chunks = chunk_markdown(SAMPLE_MD, source_path="test.md", chunk_size=64, overlap_pct=0.5)
    if len(chunks) >= 2:
        # Second chunk should have some context from the first
        assert "continued" in chunks[1].text.lower() or chunks[1].text.startswith("[")


def test_has_code_flag():
    chunks = chunk_markdown(SAMPLE_MD, source_path="test.md", chunk_size=512)
    code_chunks = [c for c in chunks if c.has_code]
    assert len(code_chunks) >= 1, "should detect code fence in markdown"


def test_empty_input():
    chunks = chunk_markdown("", source_path="empty.md")
    assert chunks == []


def test_whitespace_only_input():
    chunks = chunk_markdown("   \n\n   \n", source_path="ws.md")
    assert chunks == []