# Plugin Surface — what every agent sees

The duckbot-rag-memory project exposes the brain through several
entry points. As of **v0.14.0** they all advertise the same 12 "core
agent tools" — so an agent author can write one set of `brain_*`
calls and have it work on OpenClaw, Hermes, Codex, Cursor, and the
canonical MCP server.

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
| `python -m src.mcp_server` | **64** | MCP `tools/list` | yes (per-tool token bucket) | n/a (it's a server) |
| `scripts/duckbot-memory-mcp.sh` | 64 (wraps the MCP server) | MCP stdio | yes | n/a |
| `python -m src.extensions.duckbot_brain.adapter` | **12** | JSON-RPC over stdio (`openclaw.plugin.json`) | yes (same module) | n/a |
| `from src.plugins.memory.duckbot_brain import DuckBotBrainProvider` | **12** (function-call shape) | `plugin.yaml` | yes | `on_session_start`, `on_session_end` |
| `python -m src.cli openclaw <verb>` | **12** (shell shim, parallel to `hermes`) | argparse | yes | n/a |
| `python -m src.cli <verb>` | per-verb | argparse | n/a | n/a |
| `scripts/duckbot-ask "..."` | per-flavor (compact/snippet/json) | shell wrapper | n/a | n/a |

All three thin entry points (OpenClaw adapter, Hermes plugin, the 12
listed in the canonical MCP server as well) call the same dispatch in
`src/extensions/tools.py`. If you add a tool there, every entry point
picks it up.

## How to verify discovery from an agent

### OpenClaw (MCP stdio JSON-RPC)

Send a newline-delimited JSON-RPC request on stdin, get a response on stdout:

```bash
echo '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}' \
  | ./.venv/bin/python -m src.extensions.duckbot_brain.adapter
```

The response's `result.tools[*].name` array should be exactly:

```
brain_wake_up, brain_recall, brain_recall_verbatim, brain_remember,
brain_reflect, brain_stats, brain_fsrs_review, brain_decay_status,
brain_search_verbatim, brain_skills_list, brain_skills_promote
```

### Hermes (Python import)

```python
from src.plugins.memory.duckbot_brain import DuckBotBrainProvider
provider = DuckBotBrainProvider()
schemas = provider.get_tool_schemas()
# 11 entries, each {"type": "function", "function": {"name", "description", "parameters"}}
```

Or check the lifecycle hooks:

```python
provider.initialize(session_id="s1", hermes_home="/tmp/h", platform="cli")
context = provider.on_session_start()   # brain_wake_up shape
# ... run the agent loop ...
provider.on_session_end(messages)        # persists durable rules as procedural
```

### MCP server (canonical, 64 tools)

```bash
./scripts/duckbot-memory-mcp.sh &
# The MCP server doesn't have a CLI; clients discover via the standard
# MCP initialize → tools/list handshake.
```

### CLI / shell wrapper (no discovery needed)

```bash
# OpenClaw-style shell shim (12-tool core surface)
python -m src.cli openclaw wake-up
python -m src.cli openclaw recall "What did we decide about cloud-only models?"
python -m src.cli openclaw tools
python -m src.cli openclaw call brain_recall '{"query": "x", "k": 3}'

# Brain-query helpers (any of these)
scripts/duckbot-ask "What did we decide about cloud-only models?"
scripts/duckbot-ask -f compact -n 3 "Duckets correction style"
scripts/brain-recall "BATMAN worker offline"
```

## The 52 tools the MCP server has but the thin entry points don't

The thin 12-tool surface is intentional — it's the portable stdio subset
that works across all three thin entry points without bringing in heavy
deps (graph, blocks, quarantine, dreaming, active-memory, etc.). The
remaining 52 tools are admin / CLI tools that an agent shouldn't be
calling at runtime:

- `brain_graph_*` (7) — knowledge-graph CRUD, cognify, reconcile
- `brain_block_*` (6) — memory-block CRUD + seed_blocks
- `brain_quarantine_*` (2) — quarantine review queue
- `brain_injection_scan` (1)
- `brain_index`, `brain_inflate`, `brain_nudge`, `brain_skill_create`,
  `brain_user_model`, `brain_palace`, `brain_optimize_fsrs`,
  `brain_apply_fsrs_w20`, `brain_fsrs_optimize_apply`,
  `brain_export`, `brain_import`, `brain_seed_demo`,
  `brain_sync`, `brain_decay_apply` (14)
- `dreaming_read`, `dreaming_cycle`, `learn`, `active_memory` (4)
- `doctor`, `watch` (2)
- Legacy un-prefixed: `remember`, `recall`, `reflect`, `forget`, `stats`,
  `fsrs_review`, `decay_status`, `forget_by_query`, `search_verbatim` (9)
- Also in MCP but not thin: `active_memory`, `brain_active_memory` (2)

(Categories above overlap; 64 MCP total − 12 thin surface = 52 MCP-exclusive.
All 64 remain available via `python -m src.mcp_server`.)

If an agent needs one of these (e.g. `brain_export` for a backup), the
right path is `python -m src.cli brain_export` (CLI) or call the
canonical MCP server (`python -m src.mcp_server`).

## What the skill files tell agents

All four skills in `skills/` (`duckbot-brain/`, `openclaw-imports/`,
`codex-imports/`, `cursor-imports/`) advertise `brain_wake_up` as the
canonical session-start call. After v0.14.0, that instruction is
finally true on every platform — the tool is on every thin entry
point's surface.

## Adding a new tool

1. Add the schema + dispatch case to `src/extensions/tools.py` (one
   TOOLS dict entry + one `if name == "..."` block in `dispatch()`).
2. If it's a 10th core-agent tool (used at runtime), also list it in
   `src/extensions/duckbot_brain/openclaw.plugin.json` under `tools[]`
   so OpenClaw clients that read the manifest get it.
3. If it has a Hermes-specific shape, also add the function-call
   wrapper to `function_call_schemas()` (it inherits from TOOLS so
   this is usually free).
4. Add tests in `tests/test_extensions_tools.py`.
5. Bump the manifest `version` field (`openclaw.plugin.json`,
   `plugin.yaml`).
6. Update CHANGELOG.md.