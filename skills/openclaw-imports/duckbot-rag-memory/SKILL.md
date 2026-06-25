---
name: duckbot-rag-memory
description: Persistent RAG + memory layer for OpenClaw agents. Use brain_wake_up at session start (one-call context load), brain_inflate to recall relevant memories on demand, brain_remember to save new facts, and brain_sync to write memories back to ~/.openclaw/workspace/memory/. Powered by duckbot-rag-memory — local-first, no API costs.
metadata:
  {"openclaw": {"emoji": "🧠", "requires": {"bins": ["python"]}}}
---

# 🧠 DuckBot RAG Memory (OpenClaw Skill)

Drop-in persistent memory for OpenClaw. The enhanced brain is built on
duckbot-rag-memory: 4-tier memory model (working/episodic/semantic/
procedural), hybrid retrieval (vector + BM25 + RRF), and a one-call
wake-up that loads everything you need at session start.

## When to Use

### `/brain-wake-up` — Session start (call this FIRST every session)

`brain_wake_up` returns in one MCP call:
- top-k recent memories (filtered to drop superseded ones)
- active memory blocks (the bits the user wants you to always know)
- graph summary (top entities + recent activity)
- FSRS review queue (chunks due for re-memory)
- brief store stats

Drop-in replacement for re-reading context files every time.

### `/brain-remember` — Save what just happened

Call after meaningful events: a decision was made, the user told you
something important, you learned how a system works. The brain
auto-chunks long text, classifies tier, extracts entities, embeds,
and stores. Conflict detection: near-duplicates get the old one
marked `superseded_by` the new one.

### `/brain-recall` — Pull relevant memories

Hybrid retrieval: vector + keyword + RRF, with optional temporal-proximity
boost (newer memories score higher) and keyword boost (exact matches
get a precision bonus). Returns `chunk_id`, `text`, `tier`, `score`,
`source_path`.

### `/brain-inflate` — Formatted context block

Returns a ready-to-paste markdown block with tier labels, importance
scores, and source attribution. Use when you want to surface memories
into your context window directly.

### `/brain-sync` — Write back to context files

After significant work, write memories back to OpenClaw's workspace:
`~/.openclaw/workspace/memory/MEMORY.md`, `USER.md`, `SOUL.md`.

## MCP Tools Reference

| Tool | Purpose |
|---|---|
| `brain_wake_up` | Session-start context load (one call) |
| `brain_remember` | Save a new memory |
| `brain_recall` | Hybrid retrieval |
| `brain_inflate` | Formatted context block for injection |
| `brain_sync` | Write back to OpenClaw context files |
| `brain_reflect` | Sleep-time consolidation (episodic → semantic) |
| `brain_decay_status` | Ebbinghaus decay status per tier |
| `brain_fsrs_review` | FSRS-6 review queue |
| `brain_dreaming_cycle` | Distill episodic to semantic |
| `brain_quarantine_list` | Show quarantined injection attempts |
| `brain_stats` | Store stats |
| `brain_block_*` | Memory block read/write/list/edit |

## Install

```bash
# Add as MCP server
hermes mcp add duckbot-memory \
  --command "$HOME/Desktop/duckbot-rag-memory/scripts/duckbot-memory-mcp.sh"
# or for OpenClaw: ~/.openclaw/openclaw.json under mcp.servers.duckbot-memory
```

Then call `brain_wake_up` on session start. Cost: zero API calls.