---
name: duckbot-rag-memory
description: Persistent RAG + memory layer for Cursor. Use brain_wake_up at session start (one-call context load), brain_inflate to recall relevant memories on demand, brain_remember to save new facts. Powered by duckbot-rag-memory — local-first, no API costs.
metadata:
  {"cursor": {"emoji": "🧠", "requires": {"bins": ["python"]}}}
---

# 🧠 DuckBot RAG Memory (Cursor Skill)

Drop-in persistent memory for Cursor. The enhanced brain is built on
duckbot-rag-memory: 4-tier memory model (working/episodic/semantic/
procedural), hybrid retrieval (vector + BM25 + RRF), and a one-call
wake-up that loads everything you need at session start.

## When to Use

### `/brain-wake-up` — On-demand context (do this every session)

Call `brain_wake_up` when:
- You start a new session or task
- The user asks "what do you know about X?"
- You want to recall past decisions, patterns, or facts relevant to current work
- After ingesting new information, to understand what was just learned

**Think of it as "loading your memory before thinking."**

## MCP Tools Available

The brain exposes 63 tools via the `duckbot-memory` MCP server. The
most useful ones for Cursor sessions:

| Tool | Purpose |
|---|---|
| `brain_wake_up` | Session-start context load (one call) |
| `brain_recall` | Hybrid retrieval (vector + BM25 + RRF) |
| `brain_remember` | Save a new memory (auto-chunked, auto-tiered) |
| `brain_inflate` | Formatted context block for direct injection |
| `brain_nudge` | Proactive stale-memory surfacing |
| `brain_palace` | Wing/Room/Drawer 2D view (project-scoped recall) |
| `brain_index` | Whole-corpus AAAK scan (<500 tokens for 5000 entries) |
| `brain_sync` | Write back to OpenClaw/Hermes context files |

## Install (Cursor)

Cursor uses the same MCP JSON shape as OpenClaw/Hermes. Edit
`~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "duckbot-memory": {
      "command": "/Users/you/duckbot-rag-memory/.venv/bin/python",
      "args": ["-m", "src.mcp_server"]
    }
  }
}
```

Or use the cross-platform launcher:

```bash
hermes mcp add duckbot-memory \
  --command "$HOME/duckbot-rag-memory/scripts/duckbot-memory-mcp.sh"
```

Restart Cursor. The brain should appear in the MCP tools panel.

## First Session

In a fresh Cursor session, call `brain_wake_up`. You should get back
memories + active blocks + graph summary + FSRS review queue. If
empty, run `./scripts/openclaw-bootstrap.sh` to ingest your existing
context.

## Cost

Zero API costs. The brain runs locally (LM Studio for embeddings, no
cloud). Storage is local ChromaDB + SQLite.
