# DuckBot Brain — OpenClaw Extension

Local hybrid RAG memory for OpenClaw. **MIT, zero paid APIs.**

This extension connects the DuckBot brain project (`~/Desktop/duckbot-rag-memory`) to OpenClaw as a memory provider. OpenClaw gets 9 core tools that operate on the same brain:

- **`brain_wake_up`** — one-call session-start context load. **Call this first on every session start.** Returns recent memories (superseded filtered), active blocks, graph summary, FSRS review queue, and stats in one MCP call.
- `brain_recall` — hybrid vector + BM25 + RRF retrieval, with optional cross-encoder rerank and Ebbinghaus decay.
- `brain_recall_verbatim` — returns source bytes (never paraphrased).
- `brain_remember` — non-blocking ingest (rate-limited 10/min).
- `brain_reflect` — sleep-time episodic → semantic consolidation.
- `brain_stats` — chunk counts, graph entities, blocks, quarantine.
- `brain_fsrs_review` — chunks due for spaced-repetition review.
- `brain_decay_status` — retention scoring for recent chunks.
- `brain_search_verbatim` — exact substring match.

The 9 tools are the **core agent surface** — same list exposed by the Hermes MemoryProvider plugin (`src/plugins/memory/duckbot_brain/`) so an agent author can rely on the same tool names regardless of which platform they're on. The full 56-tool MCP surface is still available via `python -m src.mcp_server` for admin / CLI use; this thin JSON-RPC adapter is the lightweight path for runtime agent calls.

Pattern sources:
- OpenClaw's [active-memory extension](https://github.com/openclaw/openclaw/tree/main/extensions/active-memory) (TypeScript, MIT).
- The DuckBot brain's [Hermes sibling plugin](../../plugins/memory/duckbot_brain/__init__.py).

> Note: the directory is `duckbot_brain/` (underscore) because Python modules can't have hyphens, but the OpenClaw plugin id is `duckbot-brain` (hyphen, matching the rest of OpenClaw's convention).

## Install

Add to your `openclaw.json`:

```json5
{
  plugins: {
    entries: {
      "duckbot-brain": {
        enabled: true,
        config: {
          // Optional overrides (defaults shown):
          pythonPath: "~/Desktop/duckbot-rag-memory/.venv/bin/python",
          repoPath: "~/Desktop/duckbot-rag-memory",
          defaultK: 5,
          enableRerank: false,
          enableDecay: false,
          enableVerbatim: true,
          timeoutMs: 15000,
        },
      },
    },
  },
}
```

Restart the gateway. OpenClaw will discover the plugin via `openclaw.plugin.json` and launch the adapter as a stdio JSON-RPC server.

## How it works

```
OpenClaw gateway
   │  (JSON-RPC over stdio)
   ▼
adapter.py  ─── Brain.recall / Brain.recall_verbatim / Brain.remember
   │                │
   ▼                ▼
Chromadb         BGE reranker (local)
+ SQLite         LM Studio embeddings
                 Ebbinghaus decay (pure math)
```

No HTTP, no sockets, no cloud APIs. The Python adapter is invoked by OpenClaw via stdio JSON-RPC (same protocol as MCP). It imports the Brain from `src.connectors.base` — the same facade that powers the existing OpenClaw MCP server (`src/mcp_server.py`).

## Compatibility

- OpenClaw ≥ 2026-06 (any build that supports the plugin entry shape in `openclaw.plugin.json`).
- Python 3.12+ (uses `asyncio.run` for non-blocking remember).
- LM Studio running on `127.0.0.1:1234` (default) or set `LMSTUDIO_URL`.

## Layer attribution

- Layer 7 — cross-encoder rerank (`BAAI/bge-reranker-base`, MIT)
- Layer 8 — Ebbinghaus decay (public-domain math, 1885)
- Layer 13 — verbatim-first storage
- Layer 15 — pre-commit secret-scan (in this repo)
- Layer 16 — cross-runtime integration (this extension + the Hermes plugin)

## License

MIT — DuckBot brain contributors.
