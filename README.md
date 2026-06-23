# 🦆 DuckBot RAG + Memory System

> Persistent, searchable, and self-curing memory for OpenClaw + Hermes Agent.

[![Status](https://img.shields.io/badge/status-MVP-yellow)]() [![License](https://img.shields.io/badge/license-MIT-blue)]() [![Open Source](https://img.shields.io/badge/open--source-everything-green)]()

## What this is

A **RAG pipeline + memory layer** built specifically for personal AI agent usage. Inspired by (and pulling heavily from):

- **Mem0** (mem0ai/mem0) — lightweight pluggable memory layer with conflict resolution
- **Letta / MemGPT** (letta-ai/letta) — tiered memory with archival/recall split
- **Cognee** (topoteretes/cognee) — graph-augmented semantic memory
- **Hermes Agent** (NousResearch/hermes-agent) — FTS5 session search + LLM summarization
- **CoALA framework** (Princeton 2023) — 4-tier memory taxonomy (working/episodic/semantic/procedural)

It is NOT another LangChain wrapper. It's a focused, runnable pipeline tuned for one user's (Duckets') daily memory churn.

## Why it exists

DuckBot's `~/.openclaw/workspace/memory/` has **126+ daily session logs** (some 30k+ chars each). Every session, the agent dumps a "memory flush" into a daily file. When Duckets asks "what did we decide about X last month?" we have two options:

1. **Load everything into context** — burns tokens, doesn't scale, hallucination risk.
2. **Search + retrieve** — pull only the relevant chunks. This is what we built.

## Architecture (one page)

```
┌───────────────────────────────────────────────────────────────┐
│                        Cron (every 90 min)                   │
│                  22:00-10:00 America/New_York                │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│                      INGEST PIPELINE                         │
│                                                               │
│  ~/.openclaw/workspace/memory/*.md   ──┐                    │
│  ~/.openclaw/workspace/MEMORY.md     ──┤                    │
│  ~/.openclaw/workspace/AGENTS.md     ──┤  markdown-aware     │
│  ~/.openclaw/workspace/SOUL.md       ──┤  chunker            │
│  ~/Desktop/<project>/docs/*.md       ──┤  (512 tok, 15% OL) │
│                                       ──┘                    │
│                              │                                 │
│                              ▼                                 │
│  ┌──────────────────────────────────────┐                     │
│  │ Tier classifier (CoALA-inspired)     │                     │
│  │  - episodic: dated session logs      │                     │
│  │  - semantic: facts, prefs, entities  │                     │
│  │  - procedural: rules, patterns       │                     │
│  │  - working: today's hot context      │                     │
│  └──────────────────────────────────────┘                     │
│                              │                                 │
│                              ▼                                 │
│  ┌──────────────────────────────────────┐                     │
│  │ Embedding (OpenAI text-embedding-3   │                     │
│  │ -small, 1536d, $0.02/1M tokens)      │                     │
│  └──────────────────────────────────────┘                     │
│                              │                                 │
│                              ▼                                 │
│  ┌──────────────────────────────────────┐                     │
│  │ ChromaDB (embedded, persistent)      │                     │
│  │  data/chroma/                        │                     │
│  │  Collections: episodic, semantic,    │                     │
│  │  procedural, working                 │                     │
│  └──────────────────────────────────────┘                     │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│                       QUERY PIPELINE                          │
│                                                               │
│  User question → Embed → Hybrid search                       │
│                  → Reciprocal Rank Fusion (RRF)               │
│                  → Rerank with cross-encoder (optional)       │
│                  → Top-K chunks → LLM context window          │
└───────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────┐
│                       EVAL PIPELINE                           │
│                                                               │
│  Daily cron runs a 20-query benchmark set                     │
│  → recall@5, recall@10, MRR, latency                          │
│  → append to data/eval/history.jsonl                          │
│  → alert if recall drops >5pp week-over-week                  │
└───────────────────────────────────────────────────────────────┘
```

## Memory tiers (CoALA)

| Tier | What goes here | Source | Storage |
|------|---------------|--------|---------|
| **Working** | Today's active session, in-flight goals | Cron + OpenClaw heartbeat | `working` collection + ephemeral file |
| **Episodic** | Session logs, dated events, decisions made | `memory/YYYY-MM-DD.md` | `episodic` collection |
| **Semantic** | Distilled facts, user prefs, entities | LLM-extracted from episodic | `semantic` collection + `MEMORY.md` |
| **Procedural** | Rules, patterns, behavioral norms | `AGENTS.md`, `SOUL.md`, project docs | `procedural` collection |

## Quick start

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Configure (set OPENAI_API_KEY)
cp .env.example .env

# 3. Ingest all current memory
python -m src.ingest --source ~/.openclaw/workspace/memory

# 4. Query
python -m src.query "What did we decide about cua-driver last week?"

# 5. Run benchmark
python -m src.eval --benchmark benchmarks/golden.jsonl

# 6. Run cron (or schedule it)
bash scripts/cron.sh
```

## File layout

```
duckbot-rag-memory/
├── README.md
├── AGENTS.md            ← for future agents / contributors
├── CHANGELOG.md
├── LICENSE
├── requirements.txt
├── .env.example
├── src/
│   ├── __init__.py
│   ├── ingest.py        ← markdown → chunks → embeddings → chroma
│   ├── query.py         ← hybrid search + RRF + rerank
│   ├── tier.py          ← CoALA tier classifier
│   ├── chunk.py         ← markdown-aware recursive chunker
│   ├── eval.py          ← recall@K, MRR, latency benchmarks
│   ├── consolidate.py   ← dream-like episodic → semantic distillation
│   └── cli.py           ← `python -m src` entrypoint
├── scripts/
│   ├── cron.sh          ← nightly ingest + eval + commit
│   ├── eval.sh          ← manual eval run
│   └── seed.py          ← seed golden benchmark queries
├── data/                ← gitignored (chroma db, eval history)
├── docs/
│   ├── ARCHITECTURE.md  ← deeper dive
│   └── RESEARCH.md      ← every open-source project we cribbed from
├── examples/
│   └── query_demo.py
├── tests/
│   ├── test_chunk.py
│   ├── test_tier.py
│   ├── test_query.py
│   └── test_consolidate.py
└── benchmarks/
    └── golden.jsonl     ← hand-curated eval queries
```

## Status

- [x] Repo scaffold
- [ ] Chunking pipeline
- [ ] Embedding + ChromaDB ingest
- [ ] Tier classifier
- [ ] Hybrid query + RRF
- [ ] Eval harness
- [ ] Consolidation (episodic → semantic)
- [ ] Cron wiring
- [ ] Integration test with OpenClaw MEMORY.md
- [ ] v0.1 release

## License

MIT