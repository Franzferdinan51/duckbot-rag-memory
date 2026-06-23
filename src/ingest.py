"""
ingest.py — orchestrates the full ingestion pipeline.

Pipeline:
  1. Find markdown files (cli.py handles path resolution)
  2. For each file: chunk_markdown() → list[Chunk]
  3. For each chunk: classify() → tier + metadata
  4. Group chunks by tier
  5. Embed all chunks in batch per tier
  6. Add to ChromaDB (idempotent via chunk.id hash)
  7. Log per-file stats + mark ingest timestamp

Idempotent: re-running on the same files just upserts the same IDs.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .chunk import Chunk, chunk_markdown, iter_markdown_files
from .embeddings import EmbeddingProvider
from .store import MemoryStore
from .tier import Tier, classify, reclassify_for_working


@dataclass
class IngestStats:
    files_processed: int = 0
    files_skipped: int = 0
    chunks_created: int = 0
    chunks_embedded: int = 0
    chunks_added: int = 0
    by_tier: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "files_processed": self.files_processed,
            "files_skipped": self.files_skipped,
            "chunks_created": self.chunks_created,
            "chunks_embedded": self.chunks_embedded,
            "chunks_added": self.chunks_added,
            "by_tier": self.by_tier,
            "errors": self.errors,
            "duration_seconds": round(self.duration_seconds, 2),
        }


async def ingest_paths(
    paths: Iterable[str],
    store: MemoryStore,
    embedder: EmbeddingProvider,
    chunk_size: int = 512,
    overlap_pct: float = 0.15,
    force_tier: Tier | None = None,
) -> IngestStats:
    """Ingest markdown files from given paths (files or directories)."""
    stats = IngestStats()
    started = time.time()

    # Phase 1: discover files + chunk
    all_chunks: list[tuple[Chunk, Tier]] = []
    for source_path, contents in iter_markdown_files(paths):
        try:
            chunks = chunk_markdown(
                contents,
                source_path=source_path,
                chunk_size=chunk_size,
                overlap_pct=overlap_pct,
            )
        except Exception as exc:
            stats.errors.append(f"chunk {source_path}: {exc}")
            stats.files_skipped += 1
            continue
        if not chunks:
            stats.files_skipped += 1
            continue
        stats.files_processed += 1
        for chunk in chunks:
            if force_tier is not None:
                tier = force_tier
            else:
                assignment = classify(chunk.source_path, chunk.text)
                # Promote today's logs to WORKING tier
                assignment = reclassify_for_working(chunk.source_path, assignment)
                tier = assignment.tier
            all_chunks.append((chunk, tier))

    stats.chunks_created = len(all_chunks)

    # Phase 2: group by tier
    by_tier: dict[Tier, list[Chunk]] = {tier: [] for tier in Tier}
    for chunk, tier in all_chunks:
        by_tier[tier].append(chunk)

    # Phase 3: embed + add per tier
    for tier, chunks in by_tier.items():
        if not chunks:
            stats.by_tier[tier.value] = 0
            continue
        try:
            embeddings = await embedder.embed([c.text for c in chunks])
        except Exception as exc:
            stats.errors.append(f"embed {tier.value}: {exc}")
            stats.by_tier[tier.value] = 0
            continue
        stats.chunks_embedded += len(embeddings)
        try:
            added = await store.add_chunks(chunks, embeddings, tier)
            stats.chunks_added += added
        except Exception as exc:
            stats.errors.append(f"add {tier.value}: {exc}")
        stats.by_tier[tier.value] = len(chunks)

    store.mark_ingested()
    stats.duration_seconds = time.time() - started
    return stats


def write_stats_jsonl(stats: IngestStats, log_path: Path | str) -> None:
    """Append ingest stats to a JSONL log file (for cron history)."""
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.time(), **stats.to_dict()}) + "\n")


# Public API
__all__ = ["ingest_paths", "IngestStats", "write_stats_jsonl"]