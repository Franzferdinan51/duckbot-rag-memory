# Changelog

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
