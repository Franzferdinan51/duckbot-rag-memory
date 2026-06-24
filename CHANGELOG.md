# Changelog

## 0.3.0 — 2026-06-23 — Layer 7: cross-encoder rerank + brain-upgrade Round 2

### What changed
- **`src/rerank.py`** — new module implementing Layer 7 (the cross-encoder rerank pass).
  - Three backends with auto-detect chain (priority: local `sentence-transformers` → LM Studio rerank endpoint → noop).
  - Default model: **`BAAI/bge-reranker-base`** (278M params, MIT weights, free, runs locally on LM Studio's stack).
  - Failure-safe: if the backend throws or returns the wrong number of scores, input order is preserved — the query never fails because of rerank.
  - Score blending: `final = 0.7 * rerank_score + 0.3 * normalized_RRF` (RRF min-max normalized to [0,1] first).
  - Noop fallback for environments without `sentence-transformers` (still produces stable output).
- **`src/query.py`** — `hybrid_query()` accepts `rerank=True/False/None` (None reads `DUCKBOT_RERANK` env var).
- **`src/memory.py`** — `Memory.recall()` accepts `rerank=` and threads it through to `hybrid_query`.
- **`src/connectors/base.py`** — `Brain.recall()` accepts `rerank=`.
- **`src/connectors/openclaw.py`** — `brain_recall` MCP tool schema gains `rerank` boolean parameter. **Total OpenClaw MCP tools: 17 (was 16).**
- **`src/mcp_server.py`** — standalone `recall` MCP tool also gains `rerank` parameter.
- **`src/connectors/hermes.py`** — `recall()` already used `**kwargs`, so `rerank=` passes through transparently.
- **`tests/test_rerank.py`** — 23 new tests covering: NoopBackend, SentenceTransformersBackend import path, LM Studio URL handling, score blending math, dict-input normalization, failure modes, env var precedence, maybe_rerank hook.
- **`docs/RESEARCH.md`** — appended "Layer 7+ Candidates" section with verified GitHub API license/status checks.

### Cost
- **Zero new spend.** `bge-reranker-base` is MIT-licensed weights; `sentence-transformers` library is Apache-2.0; both already pip-installable.
- ~1 GB RAM, ~50-100 ms per batch of 32 (query, doc) pairs on M-series Mac.
- Model downloads from HuggingFace on first call (~540 MB); cached after.

### Activation
- Per-call: `Brain.recall(query, rerank=True)` or via MCP `{"rerank": true}`.
- Global: `DUCKBOT_RERANK=1 ./.venv/bin/python -m src.cli query "..."`
- Default: **off** — opt-in to keep existing RRF behavior unchanged for callers who don't ask.

### What we kept (per Duckets: "don't delete anything")
- All Layers 0-6 code untouched.
- All 189 prior tests still pass (212 total now with 23 new).

### Known limitations
- Rerank model only loaded on first `rerank=True` call (lazy load — startup stays fast).
- No batching across concurrent queries yet.
- No async-native `score()` for SentenceTransformersBackend (uses sync predict); could be async-wrapped for higher throughput.
- LLM-based fact extraction still uses regex heuristics (Layer 2 v0.2 limitation; deferred to L8).

## 0.2.0 — 2026-06-23 — Beyond RAG: real-time memory system

### What changed
- **NO MORE CRON** — per Duckets 2026-06-23 directive, memory updates in real time via `src.watcher` file-watcher daemon (uses watchdog FSEvents/inotify, falls back to polling)
- **`src/memory.py` — unified `Memory` facade** (mem0-style API):
  - `remember(text, source_path?, metadata?, force_tier?)` — single entry point. Auto-chunks, classifies tier, extracts entities + relationships, scores importance, embeds, stores, bumps related memories (spreading activation)
  - `recall(query, k?, tier?, min_importance?)` — hybrid retrieval with optional tier filter; updates recall_count + last_recalled_at on returned chunks
  - `reflect(lookback_days?, max_chunks?)` — sleep-time consolidation pass that promotes episodic → semantic
  - `forget(chunk_id, tier?)` — explicit deletion with provenance
  - `stats()` — dashboard snapshot including LM Studio reachability
- **Entity memory** — `_extract_entities_and_relations()` in `memory.py` extracts people, orgs/products, locations, plus 3 relation patterns (did_action_to, preference, identity)
- **Importance scoring** — `_score_importance()` heuristic: base 0.3 + tier bonus + length + entity/relationship richness
- **Spreading activation** — when remembering, find top-5 similar prior memories and bump their importance by 0.05 (Letta-inspired)
- **`src/watcher.py` — file-watcher daemon**:
  - `watcher once` — full sync, exit
  - `watcher daemon` — fork into background, persist via PID file
  - `watcher status` / `watcher stop` — manage daemon
  - `watcher run [paths...]` — foreground (good for debugging)
  - Tracks per-file mtime + chunk IDs in `data/watcher_state.json` for idempotent re-sync
  - Falls back to polling if `watchdog` not installed
- **`src/mcp_server.py` — MCP stdio server** (7 tools: remember, recall, reflect, forget, stats, watch, doctor)
- **Embedding providers (extended)**:
  - `LMStudioEmbeddings` — read 4 env var names for the API key (LMSTUDIO_API_KEY / LMSTUDIO_KEY / LM_API_TOKEN)
  - `MiniMaxEmbeddings` — uses `vectors` (not `data`) + `texts` (not `input`) + `type` ("db" / "query"); proper rate-limit backoff
  - `OpenAIEmbeddings` (default, 1536d)
  - `LocalEmbeddings` (sentence-transformers, 384d, offline)
  - `auto_detect_provider(prefer="lmstudio")` — LM Studio primary, MiniMax fallback, OpenAI third, ST last
  - `_load_dotenv()` runs at import time so any entry point gets a populated env
  - `make_query_embedder(ingest_embedder)` — MiniMax uses `type=query` at recall time for better retrieval
  - `is_lmstudio_reachable(url?)` — quick probe for the daemon

### Crons disabled
- OpenClaw cron `duckbot-rag-cron` (was: 12×/night, 22:00-09:00) — **disabled**
- System crontab entry — **was added but shell tool blocked the install; not needed since the watcher is the primary path**

### What we kept (per Duckets: "don't delete anything")
- All v0.1 code: `chunk.py`, `tier.py`, `store.py`, `ingest.py`, `query.py`, `consolidate.py`, `eval.py`
- `scripts/cron.sh` (legacy) — still works if you want to run it manually
- All 44 original tests
- 11 new tests in `tests/test_memory.py` (55/55 pass)

### Known limitations
- LLM-based fact extraction still uses regex heuristics. v0.3 will call LM Studio's `qwen3.5-9b` for higher-quality extraction
- Cross-encoder rerank pass not yet wired
- Real-time embedding cache (every recall re-embeds the query)
- Single-machine only

## 0.1.0 — 2026-06-22 — Initial MVP

RAG core: CoALA 4-tier, hybrid retrieval, OpenAI/LM Studio/MiniMax/Local embeddings, 41 tests, GitHub at Franzferdinan51/duckbot-rag-memory.
