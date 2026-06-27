---
name: duckbot-brain
description: Persistent brain for DuckBot — brain_wake_up on session start, brain_remember mid-conversation, brain_sync after changes, hourly cron auto-flush.
metadata: {"openclaw": {"emoji": "🧠", "requires": {"bins": ["python"]}}}
---

# 🧠 DuckBot Enhanced Brain

This skill gives you a persistent, searchable memory that goes beyond simple storage — it actively maintains your context so you don't start each session from a blank slate.

## Architecture

The enhanced brain is built on duckbot-rag-memory:
- **4-tier memory model** (CoALA): working / episodic / semantic / procedural
- **Hybrid retrieval**: vector search + keyword search + rank fusion
- **12-tool core surface** (`brain_wake_up` is the canonical session-start call)
- Stored memories surface via `brain_wake_up` / `brain_recall`; context files sync via `brain_sync` (MCP server only)
- **Agent-driven skill pipeline**: stamp candidates while you work, promote them to full SKILL.md files later

## When to Use

### Session start (CALL THIS FIRST every session)

`brain_wake_up` returns in ONE call: top-k recent memories, active blocks, graph summary, FSRS review queue, stats.

### Mid-conversation memory (call during, not end of session)

`brain_remember` — call after meaningful events. Auto-chunks, classifies tier, extracts entities, embeds. Non-blocking, rate-limited 10/min.

### Pull relevant memories

`brain_recall` — hybrid retrieval: vector + keyword + RRF. Returns `chunk_id`, `text`, `tier`, `score`.

### Session end / significant work complete

`brain_sync` — write memories back to workspace files. Call via `python -m src.cli sync`.

## MCP Server

`~/Desktop/duckbot-rag-memory/`. Connect via OpenClaw plugin (recommended), symlinked skill, or direct MCP.

---

## 👥 For Other Agents

**Q: Do I need to call `brain_sync` manually?**
No. An hourly cron runs automatically every hour on the 5s, calling `brain_sync --target both`. It writes memories to workspace files silently with no chat output.

**Q: Why does the cron exist?**
The brain is the authoritative store. Workspace files are a secondary mirror. The cron ensures the mirror stays updated even if no agent calls `brain_sync` during a session.

**Q: Should I still call `brain_sync` during sessions?**
Yes — as a best practice. Call `brain_sync` after meaningful work. `brain_remember` mid-conversation is more important than end-of-session sync.

**Q: What about GitHub?**
The cron does NOT push to GitHub. It only writes to local workspace files. Duckets manages what gets pushed to `Franzferdinan51/duckbot-rag-memory`.
