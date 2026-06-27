# DuckBot Brain — generic JSON-RPC MCP adapter

A stdio JSON-RPC adapter for the DuckBot brain. Use this for clients
that take an MCP server command (Claude Code, Cursor, Codex CLI,
mcporter, etc.) — **not for OpenClaw**.

OpenClaw users should install the native Node.js plugin at
[`extensions/duckbot-memory/`](../../../extensions/duckbot-memory/)
instead. That plugin spawns the Python MCP server (`src/mcp_server.py`)
as a subprocess; OpenClaw plugins run in-process in the Node gateway
and can't load Python.

This adapter exposes the 12-tool core agent surface (same list the
Hermes MemoryProvider plugin advertises) so an agent author can rely
on the same tool names regardless of which platform they're on. The
full 67 tools are available via `python -m src.mcp_server`.

The 12 tools:

- **`brain_wake_up`** — one-call session-start context load. Call this first on every session start. Returns recent memories (superseded filtered), active blocks, graph summary, FSRS review queue, and stats in one MCP call.
- `brain_recall` — hybrid vector + BM25 + RRF retrieval, with optional cross-encoder rerank and Ebbinghaus decay.
- `brain_recall_verbatim` — returns source bytes (never paraphrased).
- `brain_remember` — non-blocking ingest (rate-limited 10/min). Pass `kind="skill_candidate"` to stamp a lightweight skill candidate (no LLM).
- `brain_reflect` — sleep-time episodic → semantic consolidation.
- `brain_stats` — chunk counts, graph entities, blocks, quarantine.
- `brain_fsrs_review` — chunks due for spaced-repetition review.
- `brain_decay_status` — retention scoring for recent chunks.
- `brain_search_verbatim` — exact substring match.
- `brain_skills_list` — list unpromoted skill candidates (agent-driven pipeline).
- `brain_skills_suggest` — semantic top-N skill candidates by query.
- `brain_skills_promote` — promote a candidate to a full SKILL.md. The AGENT authors the content; the brain is pure storage + template.

## Install

### Claude Code

Add to `~/.claude.json` (or `~/.config/claude/mcp.json`):

```json
{
  "mcpServers": {
    "duckbot-memory": {
      "command": "/Users/you/duckbot-rag-memory/.venv/bin/python",
      "args": ["-m", "src.extensions.duckbot_brain.adapter"],
      "env": {
        "PYTHONPATH": "/Users/you/duckbot-rag-memory"
      }
    }
  }
}
```

### Cursor

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "duckbot-memory": {
      "command": "/Users/you/duckbot-rag-memory/.venv/bin/python",
      "args": ["-m", "src.extensions.duckbot_brain.adapter"]
    }
  }
}
```

### Codex CLI

Same shape as Claude Code, in `~/.codex/mcp.json`.

### mcporter

```bash
mcporter mcp add duckbot-memory \
  --command "$HOME/duckbot-rag-memory/.venv/bin/python" \
  --args "-m src.extensions.duckbot_brain.adapter"
```

## How it works

```
MCP client (Claude Code / Cursor / Codex / mcporter)
   │  (JSON-RPC over stdio, Content-Length framed per MCP spec)
   ▼
adapter.py  ─── Brain.recall / Brain.recall_verbatim / Brain.remember
   │                │
   ▼                ▼
ChromaDB          Qwen3 reranker (local)
+ SQLite          LM Studio embeddings
                  Ebbinghaus decay (pure math)
```

No HTTP, no sockets, no cloud APIs. The Python adapter speaks JSON-RPC
over stdio (same wire format as MCP). It imports the Brain from
`src.connectors.base` — the same facade that powers the canonical MCP
server (`src/mcp_server.py`).

## Compatibility

- Python 3.12+ (uses `asyncio.run` for non-blocking remember).
- LM Studio running on `127.0.0.1:1234` (default) or set `LMSTUDIO_URL`.
- For the default local path, load `text-embedding-embeddinggemma-300m` and `qwen3-reranker-0.6b` in LM Studio.
- Any MCP-compatible client (Claude Code, Cursor, Codex, mcporter, ...).

## Layer attribution

- Layer 7 — cross-encoder rerank (`qwen3-reranker-0.6b`, local)
- Layer 8 — Ebbinghaus decay (public-domain math, 1885)
- Layer 13 — verbatim-first storage
- Layer 15 — pre-commit secret-scan (in this repo)
- Layer 16 — cross-runtime integration (this adapter + the Hermes plugin + the OpenClaw Node.js shim)

## License

MIT — DuckBot brain contributors.
