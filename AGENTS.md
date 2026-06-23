# AGENTS.md - DuckBot RAG + Memory System

## What this is

A persistent RAG (Retrieval-Augmented Generation) + memory system built for DuckBot's OpenClaw + Hermes workflows. Inspired by (and pulling from) mem0, Letta, Cognee, Hermes Agent, and the CoALA paper.

## Quick start for agents

```bash
# Install deps
cd ~/Desktop/duckbot-rag-memory
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env to set OPENAI_API_KEY

# Ingest current memory
python -m src.cli ingest ~/.openclaw/workspace/memory

# Query
python -m src.cli query "What did we decide about cloud-only models?"

# Stats
python -m src.cli stats

# Run benchmark eval
python -m src.cli eval benchmarks/golden.jsonl

# Doctor
python -m src.cli doctor

# Run tests
pytest -v
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the deep dive.

**TL;DR:**
- 4-tier memory model (working / episodic / semantic / procedural) from the CoALA paper
- Recursive markdown-aware chunking (512 tok, 15% overlap)
- OpenAI `text-embedding-3-small` for embeddings (1536d, $0.02/1M tok)
- ChromaDB embedded for vector storage
- Hybrid retrieval: vector + BM25-style keyword search + Reciprocal Rank Fusion
- Eval harness: recall@K, MRR, p50/p95 latency
- Cron: every ~90 min from 22:00 to 10:00 EDT (see `scripts/cron.sh`)

## File layout

```
duckbot-rag-memory/
|-- src/                # core code
|   |-- chunk.py        # markdown chunker
|   |-- tier.py         # CoALA tier classifier
|   |-- embeddings.py   # pluggable embedding providers
|   |-- store.py        # ChromaDB wrapper (one collection per tier)
|   |-- ingest.py       # chunk -> tier -> embed -> upsert
|   |-- query.py        # hybrid vector + BM25 + RRF
|   |-- consolidate.py  # episodic -> semantic distillation
|   |-- eval.py         # benchmark runner
|   `-- cli.py          # python -m src.cli
|-- tests/              # 41 unit tests (pytest)
|-- benchmarks/         # golden.jsonl for eval
|-- scripts/            # cron.sh, eval.sh
|-- docs/               # ARCHITECTURE.md, RESEARCH.md
|-- data/               # gitignored: chroma db + history jsonl
|-- README.md           # one-pager
|-- CHANGELOG.md        # version history
|-- LICENSE             # MIT
`-- AGENTS.md           # this file
```

## Cron schedule

The OpenClaw cron entry (added 2026-06-22):

```
0 22-23,0-9 * * *  bash /Users/duckets/Desktop/duckbot-rag-memory/scripts/cron.sh
```

That's 12 invocations: 22:00, 23:00, 00:00-09:00. The script handles:
1. Ingest from `~/.openclaw/workspace/memory` + project docs
2. Consolidate episodic -> semantic (heuristic)
3. Run eval against `benchmarks/golden.jsonl`
4. Snapshot stats
5. Commit logs + benchmarks to local repo

Logs go to `data/logs/cron-YYYYMMDD-HHMMSS.log`.

## Design constraints

- **Idempotent ingest.** Re-running on the same file produces the same chunk IDs (content hash).
- **Per-tier storage.** Working tier can be aged out; procedural never is.
- **Open-source first.** Every external dep + every design pattern traces back to a public project (see `docs/RESEARCH.md`).
- **No agent runtime.** This is just memory, not a replacement for OpenClaw or Hermes.
- **Honest limitations.** Documented in CHANGELOG.md and inline TODO comments.

## Integration with OpenClaw

- The cron is wired via OpenClaw `cron_create` (see `scripts/install.sh` in OpenClaw).
- Ingest sources include `~/.openclaw/workspace/MEMORY.md`, `AGENTS.md`, `SOUL.md`, `memory/`.
- The `/goal` skill could call `python -m src.cli query "..."` to surface relevant prior context.

## Adding a new feature

1. Write code in `src/<module>.py` (or create a new module).
2. Add unit tests in `tests/test_<module>.py`.
3. Update `docs/ARCHITECTURE.md` if design changed.
4. Update `CHANGELOG.md` with the version + summary.
5. Commit + push to `origin/main`.

## Testing

```bash
pytest -v                  # all 41 tests
pytest tests/test_chunk.py # one module
pytest -k "tier"           # filter by name
```

Coverage gaps (future work):
- LLM-based fact extraction (currently regex-only)
- Cross-encoder rerank pass
- Concurrent ingest safety (multiple writers)
- Eval history trend detection

## Cost tracking

| Operation | Cost | Frequency |
|-----------|------|-----------|
| Embed a chunk (1536d, ~500 tok) | $0.00001 | per ingest |
| Embed 50k chunks | ~$0.50 | per cron run |
| 12 cron runs/day | ~$6/day | daily |
| Monthly (365 cron runs) | ~$180/month | monthly |

Local embedding mode cuts this to zero but is ~5x slower.

## License

MIT. See LICENSE.
