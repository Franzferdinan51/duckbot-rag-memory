# Agent Usage Recipes — DuckBot RAG memory from inside a Hermes session

This is the **agent-facing cheatsheet**: how an agent in a live Hermes
session should actually USE the brain. Not the developer recipe
(see `setup-on-windows.md` for that).

The brain is registered as MCP server `duckbot-memory` with 45 tools.
When the session starts, those tools are available as `mcp__duckbot_memory__*`
prefixed names — they show up in the agent's tool list, but the agent has
to actually *call* them for them to help.

## TL;DR

In any session where the user might need prior context:

1. **At session start**, if the user references past work, prior
   decisions, or earlier in the conversation: call
   `mcp__duckbot_memory__recall` with the relevant query before
   answering.
2. **When making a procedural rule or learning something durable**
   (user corrections, project conventions, environment facts), call
   `mcp__duckbot_memory__remember` to save it as procedural/semantic
   tier so future sessions see it.
3. **For shell-side queries** (cron jobs, scripts, ad-hoc lookups),
   the bash wrapper is `scripts/duckbot-ask` (alias: `scripts/brain-recall`).

## The 10 most-used tools

In rough order of "how often this gets called":

| Tool | When to call | Args |
|---|---|---|
| `recall` | "What did we decide about X?" / "What did the user say about Y last week?" | `query` (required), `k` (default 5), `tier` (optional, filters by tier) |
| `remember` | User states a durable preference / correction / new project fact | `text` (required), `source_path` (where it came from) |
| `stats` | "What's the brain's current state?" / Before deciding if a recall is even useful | (no args) |
| `doctor` | Something looks broken — first step is always `doctor` | (no args) |
| `recall_verbatim` | "Did the user literally say X?" — exact substring match | `query` (required), `k` (default 5) |
| `search_verbatim` | Same but lower-level (raw substring match without the full retrieval pipeline) | `needle` (required), `k` (default 5) |
| `fsrs_review` | "What should I re-surface from memory?" — items due for spaced-repetition review | `k` (default 10) |
| `decay_status` | "Is this knowledge still hot or fading?" — Ebbinghaus decay scores for recent chunks | `k` (default 20) |
| `forget_by_query` | User says "stop surfacing X" — DESTRUCTIVE | `query` (required), `k` (default 5) |
| `reflect` | "Consolidate episodic into semantic" — runs the consolidation pass | `days` (default 14) |

## When to call `recall` (the most common)

**Strong signal** — call recall before answering:
- User asks "what did we..." / "did we ever..." / "do you remember..."
- User references a past project, prior decision, or earlier session
- User says "as we discussed" / "the plan we made" / "the bug from before"
- About to repeat a workflow that might've been done before
- The user mentions a specific file/project/toolchain that has history

**Weak signal** — usually skip recall:
- Pure general knowledge question ("what's a closure in JS?")
- Live-data question ("what's the price of PRL right now?" — use
  `prl-mining` skill)
- Action request with no history component ("restart the worker")

## When to call `remember` (less common but high-value)

Save to the brain when the user states:
- A **persistent preference** ("I always want rich format for crons")
- A **correction of past behavior** ("don't ask the wallet — read the
  file")
- A **project convention** ("the local worker is X; don't ask about
  the others")
- A **fact that affects future sessions** ("alpha-miner uses LM
  Studio now, not Docker")

**Skip remember** for:
- Trivia ("I had coffee this morning") — tier classifier will route
  to episodic, fine, but not high-signal
- Live state ("PRL price is $0.61") — brain is the wrong tool, use
  live API
- Action results ("I just restarted X") — episodic auto-records this
  from session logs already

## Concrete recipe: "User asks about past decision"

```
1. mcp__duckbot_memory__recall(query="<user's topic>", k=5)
2. Read top results — particularly any with importance >= 0.7
3. Synthesize answer that references the source ("per the 2026-06-23
   morning recovery notes in alpha-miner.md, ...")
4. If the recall is empty, say so — don't fabricate
```

## Concrete recipe: "User corrects me"

```
1. Apologize briefly (one line)
2. Formulate the rule: "User wants X, not Y, because Z"
3. mcp__duckbot_memory__remember(text="<rule>", source_path="<this-session>")
4. Apply the rule going forward
5. (Optional) Confirm with the user that the rule saved
```

## Concrete recipe: "About to do something we've done before"

```
1. mcp__duckbot_memory__recall(query="<what I'm about to do>", k=3)
2. If results contain a prior recipe: use it (verbatim if exact match)
3. If results are partial: combine prior context with new info
4. If no results: tell the user "haven't done this before, going to <approach>"
```

## What NOT to do

- **Don't recall the same query 5 times in one session.** The brain has
  a built-in LRU cache (added v0.11.2). If you queried "PRL wallet"
  already, use that result.
- **Don't remember trivial events.** The watcher auto-indexes daily
  notes; you don't need to manually save "I asked about X".
- **Don't call `recall` for general knowledge.** "What's the capital
  of France?" returns garbage from the brain. Use your training
  knowledge directly.
- **Don't assume `recall` returns up-to-date data.** The brain lags
  real-time by 5+ minutes (watcher poll). For prices/weather/status,
  use live sources.

## Cross-platform notes

The shell wrappers are at:
- **macOS / Linux / git-bash**: `~/Desktop/duckbot-rag-memory/scripts/duckbot-ask`
- **Windows (PowerShell)**: `cd ~/Desktop/duckbot-rag-memory; .\scripts\duckbot-ask.ps1` (if a .ps1 wrapper exists; otherwise `bash scripts/duckbot-ask` works in git-bash)

From inside a shell:
```bash
duckbot-ask "PRL pool wallet workers"           # full JSON
duckbot-ask -f compact -n 3 "Duckets style"    # one block per result, Telegram-friendly
duckbot-ask -f snippet "BATMAN restart recipe"  # just first result, truncated
brain-recall "Duckets current mining status"    # alias
```

## Versioning

This skill tracks `Franzferdinan51/duckbot-rag-memory`. As of session
2026-06-24, the live server reports `version: 0.11.2` with 45 tools.
The next version (0.11.3+) ships `scripts/duckbot-ask` + 5-min watcher
default + 16 new tests.
