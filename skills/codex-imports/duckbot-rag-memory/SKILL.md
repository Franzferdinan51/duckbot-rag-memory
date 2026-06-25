---
name: duckbot-rag-memory
description: Persistent RAG + memory layer for OpenAI Codex CLI. Use brain_wake_up at session start (one-call context load), brain_inflate to recall relevant memories on demand, brain_remember to save new facts. Powered by duckbot-rag-memory — local-first, no API costs.
metadata:
  {"codex": {"emoji": "🧠", "requires": {"bins": ["python"]}}}
---

# 🧠 DuckBot RAG Memory (Codex CLI Skill)

Drop-in persistent memory for OpenAI Codex CLI. The enhanced brain is
built on duckbot-rag-memory: 4-tier memory model (working/episodic/
semantic/procedural), hybrid retrieval (vector + BM25 + RRF), and a
one-call wake-up that loads everything you need at session start.

## When to Use

### `/brain-wake-up` — On-demand context (do this every session)

Call `brain_wake_up` when:
- You start a new session or task
- The user asks "what do you know about X?"
- You want to recall past decisions, patterns, or facts relevant to current work

## MCP Tools Available

The brain exposes 56 tools via the `duckbot-memory` MCP server. The
most useful ones for Codex sessions:

| Tool | Purpose |
|---|---|
| `brain_wake_up` | Session-start context load (one call) |
| `brain_recall` | Hybrid retrieval |
| `brain_remember` | Save a new memory |
| `brain_inflate` | Formatted context block |
| `brain_nudge` | Proactive stale-memory surfacing |
| `brain_palace` | Wing/Room/Drawer 2D view |
| `brain_index` | Whole-corpus AAAK scan |

## Install (Codex CLI)

Codex uses the same MCP JSON shape as OpenClaw/Hermes. Edit
`~/.codex/mcp.json` (or your Codex config):

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

Restart Codex. The brain should appear in the MCP tools panel.

## First Session

In a fresh Codex session, call `brain_wake_up`. You should get back
memories + active blocks + graph summary. If empty, run
`./scripts/openclaw-bootstrap.sh` to ingest your existing context.

## Cost

Zero API costs. The brain runs locally. Storage is local ChromaDB
+ SQLite.
