# Changelog

## 0.1.0 — 2026-06-22 — Initial MVP

### What's in
- **README.md** — one-page architecture, quickstart, file layout
- **docs/RESEARCH.md** — full audit of LangChain / LlamaIndex / mem0 / Letta / Cognee / Hermes
- **docs/ARCHITECTURE.md** — deeper dive into the design
- **src/chunk.py** — markdown-aware recursive chunker (512 tok, 15% overlap, section-aware)
- **src/tier.py** — CoALA 4-tier classifier (working / episodic / semantic / procedural)
- **src/embeddings.py** — OpenAI + sentence-transformers pluggable providers
- **src/store.py** — ChromaDB wrapper with one collection per tier
- **src/ingest.py** — full ingest pipeline (chunk -> tier -> embed -> upsert)
- **src/query.py** — hybrid vector + BM25 + Reciprocal Rank Fusion
- **src/consolidate.py** — episodic -> semantic fact extraction (heuristic)
- **src/eval.py** — recall@K, MRR, latency benchmark runner
- **src/cli.py** — `python -m src.cli {ingest,query,stats,eval,consolidate,reset,doctor}`
- **scripts/cron.sh** — nightly ingest + eval + commit
- **scripts/eval.sh** — manual eval runner
- **benchmarks/golden.jsonl** — 25 hand-curated eval queries
- **tests/** — 41 unit tests across chunk / tier / query / consolidate
- **.env.example** — required env vars
- **.gitignore** — keeps data/ out of git
- **requirements.txt** — pinned dependencies
- **pytest.ini** — test config
- **LICENSE** — MIT

### Borrowed from
- LangChain `RecursiveCharacterTextSplitter` (separator order, _merge_splits) — BSD
- LlamaIndex `SentenceSplitter` — MIT
- mem0 — extraction-first approach, conflict resolution — Apache 2.0
- Letta / MemGPT — tiered memory taxonomy — Apache 2.0
- Cognee — operations vocabulary (remember/recall/improve/forget) — Apache 2.0
- Hermes Agent — FTS5 + LLM hybrid, periodic nudge pattern — MIT
- CoALA paper (Princeton 2023) — 4-tier formal model — academic

### Known limitations
- LLM-based fact extraction is not yet wired (only regex heuristics)
- Cross-encoder rerank pass is not yet wired (RRF alone for now)
- No real-time embedding cache (every query re-embeds the question)
- Single-machine only (no distributed Chroma)
- Eval benchmark is small (25 queries); needs expansion
