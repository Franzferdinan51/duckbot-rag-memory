---
name: duckbot-brain
description: DuckBot's enhanced brain — memory inflation and sync for OpenClaw and Hermes agents. Call brain_inflate when starting a new task or session, brain_sync to keep context files fresh. Powered by duckbot-rag-memory (CoALA memory model, ChromaDB, LM Studio).
metadata:
  {"openclaw": {"emoji": "🧠", "requires": {"bins": ["python"]}}}
---

# 🧠 DuckBot Enhanced Brain

This skill gives you a persistent, searchable memory that goes beyond
simple storage — it actively maintains your context so you don't start
each session from a blank slate.

## Architecture

The enhanced brain is built on duckbot-rag-memory:
- **4-tier memory model** (CoALA): working / episodic / semantic / procedural
- **Hybrid retrieval**: vector search + keyword search + rank fusion
- **Stored memories** surface via `brain_inflate`; context files sync via `brain_sync`

## When to Use

### `/brain-inflate` — On-demand context (do this every session)

Call `brain_inflate` when:
- You start a new session or task
- The user asks "what do you know about X?"
- You want to recall past decisions, patterns, or facts relevant to current work
- After ingesting new information, to understand what was just learned

**Think of it as "loading your memory before thinking."**

### `/brain-sync` — Periodic sync (cron or after major work)

Call `brain_sync` to write memories back to context files:
- `~/.openclaw/workspace/memory/MEMORY.md` — all tiers (rich markdown)
- `~/.openclaw/workspace/memory/USER.md` — user facts
- `~/.openclaw/workspace/memory/SOUL.md` — identity/principles (procedural)

This runs automatically every ~90 min via cron, but you can also trigger it
after significant work to update context files immediately.

## MCP Tools Available

### `brain_inflate`
- **query**: what you're currently working on or asking about
- **k**: max memories to return (default 10)
- **tier**: filter to working/episodic/semantic/procedural
- **min_importance**: threshold 0–1 (default 0.3)
- **agent_name**: personalize the context header

Returns a markdown block ready to paste into your thinking — tier labels,
importance bars, source attribution included.

### `brain_sync`
- **target**: `openclaw` (default) / `hermes` / `both`
- **memory_k**: max memories per tier (default 20)
- **user_k**: max user facts (default 15)
- **dry_run**: preview without writing

### Other memory tools
- `remember` — store a memory (auto-chunks, classifies tier, extracts entities)
- `recall` — hybrid retrieval with rerank and decay options
- `reflect` — consolidate episodic → semantic (periodic maintenance)
- `forget` / `forget_by_query` — selective deletion
- `stats` — brain health snapshot
- `doctor` — diagnostic check

## Memory Tiers — What Goes Where

| Tier | Content | Example |
|------|---------|---------|
| **⚡ working** | Active session, current context | "User is mid-sprint on feature X" |
| **📖 episodic** | Past experiences, events | "We tried Y in January, it failed" |
| **🧩 semantic** | Facts, concepts, knowledge | "The API uses JWT, not session cookies" |
| **⚙️ procedural** | Rules, patterns, how-to | "Run tests with `make test` before PR" |

## Pro Tips

1. **Importance scores matter** — memories with importance ≥ 0.3 appear in inflate results.
   Explicitly mark important facts when remembering: `remember("User prefers X", metadata={"importance": 0.9})`
2. **Procedural memories become your SOUL** — after reflect, high-value patterns
   migrate to procedural and sync to SOUL.md. This is how the brain builds persistent identity.
3. **User facts sync to USER.md** — things you learn about the user (preferences, habits,
   communication style) should be remembered explicitly so they appear in USER.md on next sync.
4. **Reflect runs automatically** — the cron calls `reflect` after ingest. But you can also
   call it manually: `reflect(lookback_days=7, max_chunks=200)`.

## Installation

The MCP server must be running. In OpenClaw, install the duckbot-brain skill:

```bash
# Install into your workspace skills
openclaw skills install duckbot-memory \
  --command "$HOME/Desktop/duckbot-rag-memory/scripts/duckbot-memory-mcp.sh"
```

Or symlink the skill file directly:

```bash
mkdir -p ~/.openclaw/workspace/skills/duckbot-brain/
ln -sf ~/Desktop/duckbot-rag-memory/skills/duckbot-brain/SKILL.md \
  ~/.openclaw/workspace/skills/duckbot-brain/SKILL.md
```

The skill auto-discovers from `~/.openclaw/workspace/skills/`.
