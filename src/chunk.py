"""
chunk.py — markdown-aware recursive chunker.

Based on LangChain's RecursiveCharacterTextSplitter (BSD) and LlamaIndex's
SentenceSplitter (MIT). Both projects agree the 2026 default is:
  - 400-512 tokens per chunk
  - 10-20% overlap
  - Hierarchical separator order: \n\n > \n > " " > ""

Our twist: respect markdown structure. We never break a `## Header` from its
first paragraph; we never split a fenced code block mid-line.

Why markdown-aware: DuckBot's memory is 100% markdown (MEMORY.md, daily logs,
project docs). Naive recursive splitting creates orphan headers.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable

# Rough chars-per-token heuristic for English text. OpenAI's tokenizer is
# ~4 chars/token on average. We use 3.5 to leave headroom for multi-byte chars.
CHARS_PER_TOKEN = 3.5

# Separator order matters: try the most semantic first.
# We borrow this verbatim from LangChain's RecursiveCharacterTextSplitter.
SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

# Markdown structural markers that should never be orphaned from their content.
HEADER_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
CODE_FENCE_RE = re.compile(r"^```", re.MULTILINE)
LIST_ITEM_RE = re.compile(r"^(\s*)([-*+]|\d+\.)\s+", re.MULTILINE)


@dataclass
class Chunk:
    """A single chunk with provenance metadata."""

    text: str
    source_path: str  # e.g. ~/.openclaw/workspace/memory/2026-06-22.md
    start_char: int   # offset in original document
    end_char: int
    chunk_index: int  # ordinal within the source document
    total_chunks: int  # filled in after splitting the full doc
    section_header: str | None = None  # nearest preceding ## or ###
    has_code: bool = False
    char_count: int = field(default=0)
    # L13 verbatim-first storage: preserve the original (pre-overlap, pre-prefix)
    # text so we can return the user's exact words on demand. Default = text.
    # Set explicitly during overlap application so it's distinguishable from
    # the chunk-as-displayed.
    verbatim_text: str | None = None

    def __post_init__(self) -> None:
        if self.char_count == 0:
            self.char_count = len(self.text)

    @property
    def id(self) -> str:
        """Stable ID for vector store. Uses content hash so identical chunks dedupe."""
        h = hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]
        return f"{h}-{self.chunk_index}"


def _approx_tokens(chars: int) -> int:
    return max(1, int(chars / CHARS_PER_TOKEN))


def _split_by_separator(text: str, separator: str) -> list[str]:
    """Split text by separator, keeping the separator with the preceding piece
    when it makes sense (so `## Header` stays attached to its content)."""
    if separator == "":
        return list(text)  # character-level fallback
    if separator in (". ", "? ", "! "):
        # Sentence separators: keep the punctuation
        pieces = text.split(separator)
        return [p + separator for p in pieces[:-1]] + [pieces[-1]]
    return text.split(separator)


def _find_section_header(text: str, offset: int) -> str | None:
    """Walk backward from `offset` and find the nearest preceding header."""
    # Walk backward at most 2KB; if we hit another header in that range, use it.
    window_start = max(0, offset - 2048)
    window = text[window_start:offset]
    matches = list(HEADER_RE.finditer(window))
    if matches:
        last = matches[-1]
        return last.group(0).strip()
    return None


def _contains_code_fence(text: str) -> bool:
    return bool(CODE_FENCE_RE.search(text))


def _merge(splits: list[str], separator: str, chunk_size: int) -> list[str]:
    """Merge small splits into chunks of ~chunk_size characters.

    Borrowed from LangChain's _merge_splits but tweaked to never produce
    chunks that are exactly the separator alone.
    """
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    sep_len = len(separator) if separator else 0

    for s in splits:
        s_len = len(s)
        if not s:
            continue
        # Would adding this split push us over chunk_size?
        prospective = current_len + s_len + (sep_len if current else 0)
        if current and prospective > chunk_size:
            chunks.append(separator.join(current))
            current = [s]
            current_len = s_len
        else:
            current.append(s)
            current_len = prospective

    if current:
        chunks.append(separator.join(current))
    return chunks


def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Recursively split text using the separator hierarchy."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    if not separators:
        # Fall back to character splitting (chunk_size is in chars)
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    separator = separators[0]
    splits = _split_by_separator(text, separator)
    chunks = _merge(splits, separator, chunk_size)

    # If merging produced a chunk that's still too big, recurse with next separator.
    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final.append(chunk)
        else:
            final.extend(_recursive_split(chunk, separators[1:], chunk_size))
    return final


def chunk_markdown(
    text: str,
    source_path: str = "<inline>",
    chunk_size: int = 512,
    overlap_pct: float = 0.15,
) -> list[Chunk]:
    """Split markdown text into chunks. Respects headers, lists, code fences.

    Args:
        text: The full markdown document.
        source_path: Used in chunk metadata for provenance.
        chunk_size: Target chunk size in tokens (will be converted to chars).
        overlap_pct: Fraction of chunk to overlap with neighbors (0.0-0.5).

    Returns:
        List of Chunk objects with metadata. Chunks are non-overlapping in
        `text` but the IDs are stable across re-runs of the same content.
    """
    chunk_chars = int(chunk_size * CHARS_PER_TOKEN)
    overlap_chars = int(chunk_chars * overlap_pct)

    # First pass: split into sections by ## headers. This is a structural pre-step
    # so headers stay attached to their content.
    sections: list[tuple[str, str]] = []  # (header, body)
    current_header: str | None = None
    current_body_lines: list[str] = []

    for line in text.splitlines():
        m = HEADER_RE.match(line)
        if m:
            # Flush previous section
            if current_body_lines or current_header is not None:
                sections.append((current_header or "", "\n".join(current_body_lines).strip()))
            current_header = line.strip()
            current_body_lines = []
        else:
            current_body_lines.append(line)
    # Flush last section
    if current_body_lines or current_header is not None:
        sections.append((current_header or "", "\n".join(current_body_lines).strip()))

    # If there were no headers at all, treat the whole doc as one section
    if not sections:
        sections = [(None, text)]

    chunks: list[Chunk] = []
    for section_header, body in sections:
        if not body.strip():
            continue
        # Apply recursive splitter within the section
        section_chunks = _recursive_split(body, SEPARATORS, chunk_chars)
        char_offset = text.find(body) if body in text else 0
        for idx, chunk_text in enumerate(section_chunks):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            has_code = _contains_code_fence(chunk_text)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    source_path=source_path,
                    start_char=char_offset,
                    end_char=char_offset + len(chunk_text),
                    chunk_index=len(chunks),
                    total_chunks=0,  # filled in below
                    section_header=section_header,
                    has_code=has_code,
                )
            )
            char_offset += len(chunk_text)

    # Fill in total_chunks
    for chunk in chunks:
        chunk.total_chunks = len(chunks)

    # L13 verbatim-first: snapshot the original text BEFORE we apply overlap
    # prefixes. `verbatim_text` is what we'd return for "show me exactly what
    # Duckets said" — the source bytes, not the contextualized chunk-as-stored.
    # Pattern source: MemPalace's verbatim-first design principle
    # (https://github.com/MemPalace/mempalace/blob/develop/CLAUDE.md)
    for chunk in chunks:
        if chunk.verbatim_text is None:
            chunk.verbatim_text = chunk.text

    # Apply overlap by prepending trailing sentences from the previous chunk.
    # This is "contextual retrieval" lite — preserves continuity without re-embedding.
    if overlap_chars > 0 and len(chunks) > 1:
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            tail = prev.text[-overlap_chars:].strip()
            # Don't prepend if it'd duplicate most of the chunk
            if tail and not chunks[i].text.startswith(tail[-100:]):
                # Note: we mutate `text` (the displayed chunk) but
                # `verbatim_text` stays as the pre-overlap source text.
                chunks[i].text = f"[...continued from previous section: {section_for_chunk(prev)}...]\n\n{tail}\n\n{chunks[i].text}"

    return chunks


def section_for_chunk(chunk: Chunk) -> str:
    return chunk.section_header or "(untitled section)"


def iter_markdown_files(paths: Iterable[str]) -> Iterable[tuple[str, str]]:
    """Yield (path, contents) for each markdown file under given paths.

    Paths can be files or directories (recurses).
    """
    import os
    from pathlib import Path

    for p in paths:
        path = Path(p).expanduser()
        if path.is_file() and path.suffix.lower() in {".md", ".markdown"}:
            yield (str(path), path.read_text(encoding="utf-8"))
        elif path.is_dir():
            for child in path.rglob("*"):
                if child.is_file() and child.suffix.lower() in {".md", ".markdown"}:
                    yield (str(child), child.read_text(encoding="utf-8"))