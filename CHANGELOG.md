# Changelog

## 0.4.0 — 2026-06-23 — Trust + Verbatim + Decay (L8 / L13 / L15)

MemPalace survey (verified via GitHub API 2026-06-23, MIT-licensed)
unlocked three high-value additions in one round.

### L15 — Pre-commit secret-scan + path guard

- **`scripts/secret-scan.sh`** — bash-based pre-commit hook that scans
  staged content (via `git show :<path>`) for known secret patterns:
  - OpenAI `sk-...`, Anthropic `sk-ant-...`, GitHub `ghp_...` and
    `github_pat_...`, AWS `AKIA...`, MiniMax `MiniMax-...`, Bearer tokens.
  - Private key headers (RSA, OpenSSH, PGP).
  - High-entropy generic `(api_key|secret|token|password) = "..."`.
- **Path guard**: blocks `data/chroma/`, `data/watcher_state.json`,
  `.env`, `.env.*`, `.venv/`, `node_modules/`.
- **Installed**: `.git/hooks/pre-commit` symlinked to the script.
  Also shipped as `.pre-commit-config.yaml` for users with `pre-commit`
  installed.
- **Opt-out**: `DUCKBOT_SKIP_SECRET_SCAN=1 git commit --no-verify`
  (logged to stderr).
- **17 tests** in `tests/test_secret_scan.py` covering positive cases,
  path blocks, negative cases (clean markdown, short placeholders, plain
  English with "key" or "secret"), and the opt-out env var.

### L13 — Verbatim-first storage contract

Inspired by MemPalace's verbatim-first design principle (their CLAUDE.md):
"never summarize, paraphrase, or lossy-compress user data."

- **`src/chunk.py`** — `Chunk` dataclass gains `verbatim_text` field.
  Set during `chunk_markdown()` BEFORE overlap prefixes are applied.
- **`src/store.py`** — `add_chunks` writes `verbatim_text` to Chroma
  metadata (truncated to 8 KB). Pre-L13 chunks fall back gracefully to
  `chunk.text` since no overlap was applied to them.
- **`src/connectors/base.py`** — new `Brain.recall_verbatim()` method.
  Like `recall()` but returns the source bytes, never the contextualized
  chunk. Perfect for "show me exactly what I said about X".
- **`src/connectors/openclaw.py`** — new MCP tool `brain_recall_verbatim`
  with the same query/k/tier/min_importance/rerank/decay parameters.
- **Total OpenClaw MCP tools: 19** (was 18 with v0.3.0).
- **Hermes**: `recall_verbatim()` follows the same `**kwargs` pattern
  and forwards through `Brain.recall()`.
- **14 tests** in `tests/test_verbatim.py` covering dataclass, chunk
  overlap preservation, recall_verbatim fallback, and connector schema.

### L8 — Ebbinghaus memory decay

Public-domain math: Hermann Ebbinghaus, "Memory: A Contribution to
Experimental Psychology" (1885). Retention: `R(t) = e^(-t / S)`. Same
math validated by YourMemory (+16pp LoCoMo recall vs mem0) and shipped
in MemPalace v4 as "Time-decay scoring".

- **`src/decay.py`** — new module:
  - `ebbinghaus_retention(age_days, stability_days)` — pure math.
  - `days_since(epoch)` — time-since helper.
  - `bump_stability(current_stability, recalled=True)` — stability grows
    by 1.5x per recall (FSRS-6-lite). No penalty on miss (MemPalace v4
    design choice).
  - `decay_adjust(results)` — min-max normalizes RRF scores, multiplies
    in `0.4 * retention + 0.6 * normalized_RRF`. Cold chunks (R < 0.05)
    get a 0.1x floor penalty — soft cap, never delete.
  - `maybe_decay(results, enabled=True)` — opt-in via `DUCKBOT_DECAY=1`
    env var. Default OFF (preserves existing behavior).
- **`src/query.py`** — `hybrid_query()` runs `maybe_decay()` after
  `maybe_rerank()` and before truncation. Pipeline:
  hybrid → RRF → rerank (L7) → decay (L8) → top-k.
- **`src/memory.py`** / **`src/connectors/base.py`** — `Memory.recall()`
  and `Brain.recall()` thread `decay=` through.
- **`src/connectors/openclaw.py`** / **`src/mcp_server.py`** — `recall`
  MCP schema gains `decay` boolean.
- **22 tests** in `tests/test_decay.py` covering: retention math at
  t=0 / t=S / t=2S / negative age / zero stability; days_since recency;
  bump_stability growth bound; decay_adjust cold-floor penalty; the
  `maybe_decay` opt-in env var.

### Verification

- **264/264 tests pass** (was 212 in v0.3.0; +52 new across L8/L13/L15).
- Doctor clean.
- Secret-scan tested end-to-end against real-looking leaks (caught all).
- End-to-end through `handle("brain_recall", {..., rerank: True, decay: True})`
  ranks the procedural SOUL.md rule first with full retention trace.
- End-to-end through `handle("brain_recall_verbatim", ...)` returns the
  source bytes for newly-ingested content (verbatim_text present in
  metadata, no overlap markers).

### What we kept (per Duckets: "don't delete anything")
- All Layers 0-7 code untouched. L7 still opt-in (`rerank=True`).
- All 212 prior tests still pass.
- L8 / L13 are opt-in via env vars or per-call booleans.

### What MemPalace suggested we add next (not yet shipped)
- **L14 — Pluggable backend seam** — `src/backends/base.py` with Chroma
  as default. Future-friendly for Qdrant / LanceDB without rewrites.
- **Multi-host plugin dirs** — they ship `.claude-plugin`, `.codex-plugin`,
  `.cursor-plugin`, `.antigravity-plugin` alongside MCP. We have
  OpenClaw + Hermes; the *pattern* is worth replicating.

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
