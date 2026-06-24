# DuckBot Brain — OpenClaw Extension

Local hybrid RAG memory for OpenClaw. **MIT, zero paid APIs.**

This extension connects the DuckBot brain project (`~/Desktop/duckbot-rag-memory`) to OpenClaw as a memory provider. OpenClaw gets three additional tools that operate on the same brain:

- `brain_recall` — hybrid vector + BM25 + RRF retrieval, with optional cross-encoder rerank and Ebbinghaus decay.
- `brain_recall_verbatim` — returns source bytes (never paraphrased).
- `brain_remember` — non-blocking ingest.
- `brain_stats` — chunk counts, last query.

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
