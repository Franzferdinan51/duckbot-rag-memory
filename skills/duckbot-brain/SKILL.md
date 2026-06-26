---
name: duckbot-brain
description: DuckBot's enhanced brain — persistent memory for OpenClaw and Hermes agents. Call brain_wake_up first on every session start (one-call context load), brain_recall to search, brain_remember to save. Powered by duckbot-rag-memory (CoALA memory model, ChromaDB, LM Studio).
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
- **12-tool core surface** (`brain_wake_up` is the canonical session-start call)
- Stored memories surface via `brain_wake_up` / `brain_recall`; context files sync via `brain_sync` (MCP server only)
- **Agent-driven skill pipeline**: stamp candidates while you work, promote them to full SKILL.md files later (zero VRAM on the brain side — only the embedding model runs)

## When to Use

### `/brain-wake-up` — Session start (CALL THIS FIRST every session)

`brain_wake_up` returns in ONE call:
- top-k recent memories (filtered to drop superseded chunks)
- active memory blocks (the bits the user wants you to always know)
- graph summary (top entities + recent activity)
- FSRS review queue (chunks due for re-memory)
- brief store stats

Drop-in replacement for re-reading context files every time. With
`query=""` (the default), pulls the most-recent episodic + procedural
chunks — the "what was I doing recently?" wake-up.

### `/brain-recall` — Pull relevant memories

Hybrid retrieval: vector + keyword + RRF, with optional temporal-proximity
boost (newer memories score higher) and keyword boost. Returns `chunk_id`,
`text`, `tier`, `score`, `source_path`. Pass `rerank=true` for cross-encoder
boost, `decay=true` for Ebbinghaus retention weighting.

### `/brain-remember` — Save what just happened

Call after meaningful events: a decision was made, the user told you
something important, you learned how a system works. The brain
auto-chunks long text, classifies tier, extracts entities, embeds,
and stores. Non-blocking, rate-limited 10/min.

### `/brain-sync` — Write back to context files (MCP server only)

After significant work, write memories back to OpenClaw's workspace:
`~/.openclaw/workspace/memory/MEMORY.md`, `USER.md`, `SOUL.md`. Note:
this tool is on the full MCP server (64 tools), not the thin 12-tool
agent surface — call it via `python -m src.cli sync`.

## Agent-Driven Skill Pipeline (Zero VRAM)

The brain never calls a generative LLM — only the embedding model runs.
The AGENT authors skill content using its own LLM context; the brain is
pure storage + template.

### Flow

1. **Finish a task worth repeating** → stamp a candidate:
   ```
   brain_remember(
     text="Restarted BATMAN by running docker compose down && up",
     kind="skill_candidate",
     summary="BATMAN container restart",
     importance=0.8,
   )
   ```
   The brain just stores + embeds the chunk in the procedural tier with
   `metadata.kind="skill_candidate"`. No LLM call. Returns `chunk_id`
   immediately (blocking, so you can promote it later).

2. **At a quiet moment (end of session, consolidate pass)** → review:
   ```
   brain_skills_list()
   ```
   Returns unpromoted candidates sorted by recency then importance.

3. **Write the skill yourself** (you have the full context — what went
   wrong, what worked, what the user prefers). Then promote:
   ```
   brain_skills_promote(
     chunk_id="mem_abc123",
     name="Restart BATMAN Container",
     description="Use this when BATMAN is offline and needs a restart",
     instructions=[
       "Run docker compose down in the BATMAN directory",
       "Run docker compose up -d",
       "Verify health with curl localhost:8080/health",
     ],
     example="docker compose down && docker compose up -d",
   )
   ```
   The brain writes `skills/<slug>/SKILL.md` (pure template, no LLM) and
   marks the candidate chunk as `promoted=True`.

### Why agent-driven (not LLM-driven)

A separate LLM call from the brain trying to extract a SKILL.md from a
chunk is at best a lossy summary, and at worst hallucinates. The agent
that did the task has the full context. The brain is a substrate; the
agent is the author.

## MCP Tools Available

### `brain_wake_up` (THE session-start call)
- **query**: optional anchor; blank = recent-memory wake-up
- **k**: max memories to return (default 8)
- **include_blocks** / **include_graph** / **include_fsrs_review**: toggles for the bundled sections

### `brain_recall`
- **query**: what you're currently working on or asking about
- **k**: max results (default 5)
- **tier**: filter to working/episodic/semantic/procedural
- **rerank** / **decay**: optional retrieval boosts

### `brain_remember`
- **text**: the memory content
- **source**: where it came from
- **tier**: optional override
- **kind**: pass `"skill_candidate"` to stamp a lightweight skill candidate (agent-driven pipeline, no LLM, stored in procedural tier)
- **summary** / **importance**: optional metadata for skill candidates

### Other core tools
- `brain_recall_verbatim` — original source bytes (never paraphrased)
- `brain_reflect` — consolidate episodic → semantic (periodic maintenance)
- `brain_stats` — brain health snapshot
- `brain_fsrs_review` — spaced-repetition review queue
- `brain_decay_status` — retention scoring for recent chunks
- `brain_search_verbatim` — exact substring match
- `brain_skills_list` — list unpromoted skill candidates (agent-driven pipeline)
- `brain_skills_promote` — promote a candidate to a full SKILL.md (agent authors content, brain is pure template)

## Memory Tiers — What Goes Where

| Tier | Content | Example |
|------|---------|---------|
| **⚡ working** | Active session, current context | "User is mid-sprint on feature X" |
| **📖 episodic** | Past experiences, events | "We tried Y in January, it failed" |
| **🧩 semantic** | Facts, concepts, knowledge | "The API uses JWT, not session cookies" |
| **⚙️ procedural** | Rules, patterns, how-to | "Run tests with `make test` before PR" |

## Pro Tips

1. **Always `brain_wake_up` first** — it's cheaper than 4 separate calls
   and returns everything you need to continue a previous conversation.
2. **Importance scores matter** — memories with importance ≥ 0.3 surface
   in wake-up results. Explicitly mark important facts when remembering.
3. **Procedural memories become your SOUL** — after reflect, high-value
   patterns migrate to procedural and sync to SOUL.md.
4. **Reflect runs automatically** — the cron calls `reflect` after ingest.
   But you can also call it manually: `reflect(lookback_days=7, max_chunks=200)`.

## Installation

The brain is reachable three ways — pick whichever fits your runtime:

**1. As an OpenClaw native plugin (recommended for OpenClaw users — 64 tools + auto session hooks):**

```bash
# Auto-install via the bootstrap script (recommended)
./scripts/openclaw-bootstrap.sh

# This symlinks extensions/duckbot-memory/ into ~/.openclaw/extensions/
# and prints the next step. The plugin is a pure Node.js shim — zero
# npm deps — that spawns the Python MCP server and wires session_start /
# session_end hooks so brain_wake_up fires automatically.

# After bootstrap, activate:
openclaw gateway restart
openclaw plugins list | grep duckbot-memory    # should show "✓ installed"
```

**2. As a symlinked skill (skill discovery — already done by bootstrap):**

```bash
mkdir -p ~/.openclaw/workspace/skills/duckbot-brain/
ln -sf ~/Desktop/duckbot-rag-memory/skills/duckbot-brain/SKILL.md \
  ~/.openclaw/workspace/skills/duckbot-brain/SKILL.md
```

**3. As the canonical MCP server (for Hermes, Claude Code, Cursor, Codex, or any MCP client):**

```bash
hermes mcp add duckbot-memory \
  --command "$HOME/Desktop/duckbot-rag-memory/scripts/duckbot-memory-mcp.sh"
```

See `docs/PLUGIN_SURFACE.md` for the full entry-point comparison.
