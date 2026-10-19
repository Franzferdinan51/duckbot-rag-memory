# Architecture Deep-Dive

## Why a custom RAG, not LangChain?

We considered LangChain, LlamaIndex, Haystack, RAGFlow. We rejected all of them for the same reason: **too much surface area for a focused, personal-scale pipeline.**

DuckBot's memory system has a single user (Duckets) and a single use case (recall across sessions). We don't need:
- Agent frameworks (we have OpenClaw)
- Document loaders for PDFs, DOCX, HTML scraping (we only ingest markdown)
- Streaming response handlers (cron runs offline)
- Token-by-token callbacks
- LangSmith observability (we have logs)

We DO need:
- Markdown-aware chunking that respects headers
- CoALA 4-tier separation so we can age out old working memory
- Hybrid vector + keyword search (RRF)
- Idempotent re-ingestion (same chunk.md -> same vector ID)
- Eval harness that catches regressions
- Agent-owned fact extraction with regex fallback (episodic -> semantic) — the "dream" pass

Total expected scale: 50k-100k chunks across all tiers. ChromaDB embedded mode handles this comfortably on a Mac mini.

## Chunking strategy

**Recursive character splitting** with markdown awareness.

Order of separators (try in order):
1. `\n\n` — paragraph breaks (most semantic)
2. `\n` — line breaks
3. `. `, `? `, `! ` — sentence endings (keep the punctuation)
4. ` ` — word breaks (last resort)

Target chunk size: **512 tokens** (roughly 1800 chars). This is the 2026 consensus default — large enough to hold meaningful context, small enough to fit in any embedding model's context window.

Overlap: **15%** (~77 tokens). Enough to preserve cross-chunk continuity without wasting storage.

**Markdown twist:** we never break a `## Header` from its first paragraph. The chunker pre-splits on `##` / `###` headers first, then runs the recursive splitter within each section.

## Embedding model

**Deployed (reference Mac):** `text-embedding-nomic-embed-text-v1.5`
via LM Studio, 768-dim (`DUCKBOT_EMBEDDING=lmstudio`,
`LMSTUDIO_EMBED_DIM=768`). The bank was built at 768 — **never change
the dimension source on an existing bank, or every stored vector is
invalidated.** The nightly jobs self-load the model (load-on-dispatch);
it is never left resident.

**Alternatives** (for new banks, via `DUCKBOT_EMBEDDING=`):
- OpenAI `text-embedding-3-small` (1536d, $0.02 per 1M tokens) — cheap cloud, good MTEB
- Minimax / NVIDIA NIM cloud providers — see `src/embeddings.py`
- Local `BAAI/bge-small-en-v1.5` (384d, sentence-transformers) — offline, free

## Storage layout

ChromaDB embedded, persistent at `data/chroma/`. One collection per tier:
- `duckbot_working` — today's active session, capped at 100 chunks
- `duckbot_episodic` — daily session logs, dated `YYYY-MM-DD.md`
- `duckbot_semantic` — distilled facts, user prefs, entities
- `duckbot_procedural` — rules, patterns, behavioral norms
- `_meta` — internal (last_ingest_ts, last_query_ts)

Why per-tier collections:
1. We can run tier-specific queries ("show me procedural rules")
2. Different eviction policies per tier (working tier = LRU, procedural = never)
3. Metadata schemas can differ
4. Usage stats tracked independently

## Retrieval: hybrid + RRF

Pattern from Cognee + LangChain hybrid retriever.

For each query:
1. Embed the query text (1536d vector)
2. Vector search top-N*3 across all tier collections (cosine distance)
3. BM25-style keyword search: ChromaDB's `where_document: $contains` filtered against the query's keywords
4. Reciprocal Rank Fusion (RRF): combine ranks, not scores
   - Formula: `score(d) = sum(1 / (k + rank_i(d)))` for each retriever i
   - k=60 is the standard constant (Cormack et al. 2009)
5. Sort by RRF desc, return top-N

Why RRF: it doesn't require score normalization (vector cosine distance and BM25 hits aren't on the same scale). It just uses ranks, which are scale-invariant.

**Future:** add a cross-encoder rerank pass (Cohere, Jina, or local `qwen3-reranker`) before returning final results. Skipped for v0.1 — RRF alone is good enough at our scale.

## Consolidation: episodic -> semantic

Periodically (cron-driven), we run a "dream" pass:
1. Pull recent episodic chunks (last 7 days)
2. Group by topic (cosine similarity clustering — simple threshold for v0.1)
3. Extract durable facts: agent-supplied facts, or the Hy-Memory writer (`src/hy_memory.py`, load-on-dispatch — the live nightly pass), with regex heuristics as the fail-open fallback
4. Add extracted facts to the SEMANTIC tier
5. Optionally mark old episodic chunks as superseded (skip for v0.1)

The fact extraction patterns in `consolidate.py` look for:
- `Duckets said X` / `user said X` (user-said)
- `we decided X` / `we'll X` (decision)
- `Always X` / `Never X` / `must X` (rule)
- `Installed X` / `set up X` (setup)
- `lives at X` / `address is X` (location)
- `prefers X` / `likes X` (preference)

These are filtered through a dedup pass (content-hash / Jaccard) before insertion. The regex path does not load a chat model — the agent supplies the facts or the heuristics run. The Hy-Memory nightly pass is the deliberate exception: it loads its dedicated writer on dispatch, unloads it after, and leaves the chat model and embedder untouched.

## Eval methodology

**LoCoMo-style benchmark** (the same one mem0 uses):
- Hand-curated golden queries with known-correct chunks
- Metrics: recall@5, recall@10, MRR, p50/p95 latency
- Run weekly via cron, alert on regression

Our `benchmarks/golden.jsonl` is 25 queries drawn from real DuckBot sessions, with `expected_keywords` for hit detection (any-of match).

The history file (`data/eval_history.jsonl`) lets us track recall over time. Future: detect a >5pp drop week-over-week and alert.

## Cron schedule

The legacy `scripts/cron.sh` (12 invocations across a 12-hour inactivity
window) is **deprecated as of v0.15.0** — moved to `scripts/archive/`.
The watcher daemon (`scripts/start-watcher.sh`) replaced its ingest role:
5-minute polling, content-hash dedup, logs to `data/watcher.log`.

Scheduled maintenance on the reference Mac lives in Hermes cron
(host-side, not repo code — full lineup in the README deployed-jobs
table): nightly Hy-Memory consolidation + sync (03:00, no-agent),
decay pruning (03:00, `qwen3-reranker-0.6b`), FSRS self-tune
(Sun 04:00, no-agent), hourly brain-sync, and a service guardian
(08:00/14:00/20:00).

Inactivity hours intentionally avoid the active work day (10am-10pm). This means:
- Ingests run while Duckets sleeps / works without RAG churn
- Memory churn from daytime sessions gets consolidated overnight
- Morning cron run (9-10am) has the latest state ready for Duckets's first session of the day

## Why not run continuously?

Embeddings cost money (or local RAM/CPU). Ingesting the full ~126 daily logs is ~50k chunks * ~500 tokens = ~25M tokens = ~$0.50 per run on OpenAI. The reference Mac runs local LM Studio, so the cost there is zero — cadence is bounded by RAM and model load, not dollars. The batched-cron approach is 7-8x cheaper on cloud and keeps local load predictable.

Also: cron lets us batch commits and surface alerts. Continuous ingestion would be silent.
