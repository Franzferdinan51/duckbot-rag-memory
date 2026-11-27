# Plugin Surface — what every agent sees

The duckbot-rag-memory project exposes the brain through several
entry points. They all advertise the same 12 "core agent tools" — so
an agent author can write one set of `brain_*` calls and have it work
on Hermes, Codex, Cursor, and the canonical MCP server.

> **v0.19.0:** OpenClaw support was removed at the owner's direction
> ("we don't really support OpenClaw anymore — I use you and Hermes").
> The Node.js shim (`extensions/duckbot-memory/`), the
> `python -m src.cli openclaw` shell shim, the OpenClaw connector, and
> the `skills/openclaw-imports/` manifests are gone. The generic
> JSON-RPC adapter and the Hermes plugin remain, and the MCP server
> surface is **67 tools** (the dreaming bridge and Active Memory
> aliases were removed with OpenClaw).

## The 12 core tools

| # | Tool | What it does |
|---|---|---|
| 1 | `brain_wake_up` | **Call this first on session start.** One-call context load: recent memories (superseded filtered), active blocks, graph summary, FSRS review queue, stats. |
| 2 | `brain_recall` | Hybrid retrieval (vector + BM25 + RRF). `rerank=true` for cross-encoder boost; `decay=true` for Ebbinghaus retention weighting. |
| 3 | `brain_recall_verbatim` | Returns the original (pre-overlap, pre-prefix) source text. Use for "what exactly did I say?" |
| 4 | `brain_remember` | Persist a memory. Non-blocking, rate-limited 10/min. Pass `kind="skill_candidate"` to stamp a lightweight skill candidate (agent-driven pipeline, no LLM). |
| 5 | `brain_reflect` | Sleep-time consolidation: merge episodic → semantic. Long-running; call once per cron. |
| 6 | `brain_stats` | One-glance snapshot: vector counts per tier, graph entities, block count, quarantine totals. |
| 7 | `brain_fsrs_review` | Chunks due for FSRS-6 spaced-repetition review. Public-domain math, no LLM. |
| 8 | `brain_decay_status` | Ebbinghaus retention scoring for recent chunks. Public-domain math (1885), no LLM. |
| 9 | `brain_search_verbatim` | Exact substring match against the verbatim (pre-overlap) text. |
| 10 | `brain_skills_list` | List unpromoted skill candidates (agent-driven pipeline). The agent reads these and decides which to promote. No LLM. |
| 11 | `brain_skills_suggest` | Semantic top-N skill candidates matching a query (agent-driven pipeline). Use when the agent is working on a topic and wants to know 'are there candidate skills about X?'. No LLM. |
| 12 | `brain_skills_promote` | Promote a candidate to a full SKILL.md. The AGENT authors the content; the brain is pure template. `instructions_markdown` lets the agent author full markdown (overrides the flat `instructions` list). No LLM. |

The first entry is intentional: `brain_wake_up` is the canonical
session-start call.

## Entry-point comparison

| Entry point | Tools exposed | Discovery shape | Rate-limited? | Hooks? |
|---|---|---|---|---|
| `python -m src.mcp_server` | **67** | MCP `tools/list` | yes (per-tool token bucket) | n/a (it's a server) |
| `scripts/duckbot-memory-mcp.sh` | 67 (wraps the MCP server) | MCP stdio | yes | n/a |
| `python -m src.extensions.duckbot_brain.adapter` | **12** | Generic JSON-RPC over stdio (for Claude Code / Cursor / Codex / mcporter) | yes (same module) | n/a |
| `from src.plugins.memory.duckbot_brain import DuckBotBrainProvider` | **12** (function-call shape) | `plugin.yaml` | yes | `on_session_start`, `on_session_end` |
| `python -m src.cli hermes <verb>` | **12** (shell shim) | argparse | yes | n/a |
| `python -m src.cli <verb>` | per-verb | argparse | n/a | n/a |
| `scripts/duckbot-ask "..."` | per-flavor (compact/snippet/json) | shell wrapper | n/a | n/a |

Both thin entry points (Hermes plugin, JSON-RPC adapter) call the
same dispatch in `src/extensions/tools.py`. If you add a tool to
`src/mcp_server.py`'s TOOLS list or to the thin surface, both
surfaces pick it up.

## How to verify discovery from an agent

### Generic JSON-RPC adapter

The Python adapter at `src/extensions/duckbot_brain/adapter.py`
speaks MCP stdio JSON-RPC (the 12-tool thin surface):

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' \
  | ./.venv/bin/python -m src.extensions.duckbot_brain.adapter
```

The response's `result.tools[*].name` array should be exactly:

```
brain_wake_up, brain_recall, brain_recall_verbatim, brain_remember,
brain_reflect, brain_stats, brain_fsrs_review, brain_decay_status,
brain_search_verbatim, brain_skills_list, brain_skills_suggest,
brain_skills_promote
```

### Hermes (Python import)

```python
from src.plugins.memory.duckbot_brain import DuckBotBrainProvider
provider = DuckBotBrainProvider()
schemas = provider.get_tool_schemas()
# 12 entries, each {"type": "function", "function": {"name", "description", "parameters"}}
```

Or check the lifecycle hooks:

```python
provider.initialize(session_id="s1", hermes_home="/tmp/h", platform="cli")
context = provider.on_session_start()   # brain_wake_up shape
# ... run the agent loop ...
provider.on_session_end(messages)        # persists durable rules as procedural
```

### MCP server (canonical, 67 tools)

```bash
./scripts/duckbot-memory-mcp.sh &
# The MCP server doesn't have a CLI; clients discover via the standard
# MCP initialize → tools/list handshake.
```

### CLI / shell wrapper (no discovery needed)

```bash
# Hermes shell shim (12-tool core surface)
python -m src.cli hermes wake-up
python -m src.cli hermes recall "What did we decide about cloud-only models?"
python -m src.cli hermes tools
python -m src.cli hermes call brain_recall '{"query": "x", "k": 3}'

# Brain-query helpers (any of these)
scripts/duckbot-ask "What did we decide about cloud-only models?"
scripts/duckbot-ask -f compact -n 3 "Duckets correction style"
scripts/brain-recall "BATMAN worker offline"
```

## The MCP server's tools beyond the thin 12

The thin 12-tool surface is intentional — it's the portable stdio
subset that works across the thin entry points without bringing in
heavy deps. The canonical MCP server exposes **67 tools** in total;
the remaining 55 are admin / server-side tools an agent shouldn't
normally call at runtime:

- `brain_graph_*` (7) — knowledge-graph CRUD, cognify, reconcile
- `brain_block_*` (6) — memory-block CRUD + seed_blocks
- `brain_quarantine_*` (2) — quarantine review queue
- `brain_injection_scan` (1)
- `brain_index`, `brain_inflate`, `brain_nudge`, `brain_skill_create`,
  `brain_user_model`, `brain_palace`, `brain_optimize_fsrs`,
  `brain_apply_fsrs_w20`, `brain_fsrs_optimize_apply`,
  `brain_export`, `brain_import`, `brain_seed_demo`,
  `brain_sync`, `brain_decay_apply` (14)
- `learn`, `doctor`, `watch` (3)
- The connector family registered from
  `src/connectors/brain_tools.py` (graph/blocks/quarantine/dashboard
  dispatch for names without native handlers)
- Legacy un-prefixed: `remember`, `recall`, `reflect`, `forget`,
  `stats`, `fsrs_review`, `decay_status`, `forget_by_query`,
  `search_verbatim` (9)

(Categories overlap — several tools are reachable under both a
`brain_*` name and a legacy name. The authoritative count is the
67 returned by `tools/list`.)

If an agent needs one of these (e.g. `brain_export` for a backup),
the right path is `python -m src.cli brain_export` (CLI) or call the
canonical MCP server (`python -m src.mcp_server`).

## What the skill files tell agents

All three skills in `skills/` (`duckbot-brain/`, `codex-imports/`,
`cursor-imports/`) advertise `brain_wake_up` as the canonical
session-start call. That instruction is true on every entry point —
the tool is on every thin surface.

## Adding a new tool

1. Add the schema + dispatch case to `src/extensions/tools.py` (one
   TOOLS dict entry + one `if name == "..."` block in `dispatch()`)
   AND/OR to `src/mcp_server.py`'s TOOLS list (the canonical 67-tool
   surface). Pick whichever surface the tool belongs to.
2. If it's a core-agent tool (used at runtime), no further work — the
   thin surface picks it up automatically.
3. If it has a Hermes-specific shape, also add the function-call
   wrapper to `function_call_schemas()` (it inherits from TOOLS so
   this is usually free).
4. Add tests in `tests/test_extensions_tools.py` and/or
   `tests/test_skill_pipeline.py` as appropriate.
5. Bump the manifest `version` fields
   (`src/plugins/memory/duckbot_brain/plugin.yaml` and the
   `hermes/plugins/` copy — they must stay byte-identical).
6. Update CHANGELOG.md.
