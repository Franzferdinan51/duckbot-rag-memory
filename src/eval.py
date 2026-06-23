"""
eval.py — retrieval quality eval harness.

Runs a benchmark JSONL of {query, expected_tier?, expected_keywords?} entries
against the current store and reports recall@K, MRR, latency.

This is the "is my RAG still working" sanity check. We run it weekly via cron
and alert if recall drops >5pp week-over-week.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .embeddings import get_default_provider
from .query import hybrid_query, QueryStats
from .store import MemoryStore


@dataclass
class EvalEntry:
    query: str
    expected_keywords: list[str] = field(default_factory=list)  # any-of match
    expected_tier: str | None = None
    expected_source_path: str | None = None
    expected_section: str | None = None


@dataclass
class EvalSampleResult:
    query: str
    recall_at_5: float
    recall_at_10: float
    mrr: float
    first_hit_rank: int | None
    latency_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "recall_at_5": self.recall_at_5,
            "recall_at_10": self.recall_at_10,
            "mrr": self.mrr,
            "first_hit_rank": self.first_hit_rank,
            "latency_seconds": round(self.latency_seconds, 3),
        }


@dataclass
class EvalSummary:
    samples: list[EvalSampleResult]
    mean_recall_at_5: float
    mean_recall_at_10: float
    mean_mrr: float
    p50_latency: float
    p95_latency: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": [s.to_dict() for s in self.samples],
            "summary": {
                "mean_recall_at_5": round(self.mean_recall_at_5, 3),
                "mean_recall_at_10": round(self.mean_recall_at_10, 3),
                "mean_mrr": round(self.mean_mrr, 3),
                "p50_latency_seconds": round(self.p50_latency, 3),
                "p95_latency_seconds": round(self.p95_latency, 3),
                "n_samples": len(self.samples),
            },
        }


def load_benchmark(path: Path | str) -> list[EvalEntry]:
    """Load benchmark JSONL. One EvalEntry per line."""
    p = Path(path)
    entries: list[EvalEntry] = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            data = json.loads(line)
            entries.append(EvalEntry(
                query=data["query"],
                expected_keywords=data.get("expected_keywords", []),
                expected_tier=data.get("expected_tier"),
                expected_source_path=data.get("expected_source_path"),
                expected_section=data.get("expected_section"),
            ))
    return entries


def _is_hit(result_text: str, result_meta: dict, entry: EvalEntry) -> bool:
    """Decide if a single search result counts as a hit for this eval entry."""
    text_lower = result_text.lower()
    # Keyword match (any-of)
    if entry.expected_keywords:
        return any(k.lower() in text_lower for k in entry.expected_keywords)
    # Tier match
    if entry.expected_tier and result_meta.get("tier") != entry.expected_tier:
        return False
    # Source path match
    if entry.expected_source_path:
        if entry.expected_source_path not in result_meta.get("source_path", ""):
            return False
    # Section header match
    if entry.expected_section:
        section = result_meta.get("section_header") or ""
        if entry.expected_section.lower() not in section.lower():
            return False
    # If no criteria specified, count as hit (useful for latency-only eval)
    return True


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    k = (len(sorted_v) - 1) * pct / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_v) - 1)
    if f == c:
        return sorted_v[f]
    return sorted_v[f] * (c - k) + sorted_v[c] * (k - f)


async def run_eval(
    benchmark_path: Path | str,
    store: MemoryStore | None = None,
    embedder=None,
    n_results: int = 10,
) -> EvalSummary:
    """Run the benchmark and compute summary metrics."""
    if store is None:
        store = MemoryStore()
    if embedder is None:
        embedder = get_default_provider()
    entries = load_benchmark(benchmark_path)
    sample_results: list[EvalSampleResult] = []

    for entry in entries:
        results, stats = await hybrid_query(
            entry.query, store, embedder, n_results=n_results
        )
        # Determine first hit rank
        first_hit_rank = None
        for i, r in enumerate(results, 1):
            if _is_hit(r.text, r.metadata, entry):
                first_hit_rank = i
                break
        recall5 = 1.0 if first_hit_rank is not None and first_hit_rank <= 5 else 0.0
        recall10 = 1.0 if first_hit_rank is not None and first_hit_rank <= 10 else 0.0
        mrr = 0.0 if first_hit_rank is None else 1.0 / first_hit_rank
        sample_results.append(EvalSampleResult(
            query=entry.query,
            recall_at_5=recall5,
            recall_at_10=recall10,
            mrr=mrr,
            first_hit_rank=first_hit_rank,
            latency_seconds=stats.duration_seconds,
        ))

    latencies = [s.latency_seconds for s in sample_results]
    summary = EvalSummary(
        samples=sample_results,
        mean_recall_at_5=sum(s.recall_at_5 for s in sample_results) / max(1, len(sample_results)),
        mean_recall_at_10=sum(s.recall_at_10 for s in sample_results) / max(1, len(sample_results)),
        mean_mrr=sum(s.mrr for s in sample_results) / max(1, len(sample_results)),
        p50_latency=_percentile(latencies, 50),
        p95_latency=_percentile(latencies, 95),
    )
    return summary


def append_history(summary: EvalSummary, history_path: Path | str) -> None:
    """Append eval summary to history JSONL for trend tracking."""
    p = Path(history_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps({
            "ts": time.time(),
            **summary.to_dict()["summary"],
        }) + "\n")


__all__ = [
    "EvalEntry",
    "EvalSummary",
    "load_benchmark",
    "run_eval",
    "append_history",
]