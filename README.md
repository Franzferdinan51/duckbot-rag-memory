# DuckBot RAG + Memory System

Persistent, searchable memory that dramatically expands the limited default memory of OpenClaw and Hermes Agent. Drop-in replacement for the flat chat-history buffers both agents ship with — adds a 4-tier CoALA memory model, hybrid retrieval, entity graph, verbatim recall, FSRS-6 spaced repetition, dreaming consolidation, and a Wing/Room/Drawer 2D hierarchy on top.

[![Status](https://img.shields.io/badge/latest_changelog-0.15.1-yellow)]()
[![MCP](https://img.shields.io/badge/MCP_server-0.15.1-green)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![CI](https://github.com/Franzferdinan51/duckbot-rag-memory/actions/workflows/ci.yml/badge.svg)]()

## What This Is

DuckBot memory is a focused RAG and long-term memory layer for personal agent workflows. It ingests markdown, classifies it into memory tiers, embeds it, stores it in local vector collections, and exposes recall/write tools through CLI wrappers and MCP.

## What You Get That Default Memory Doesn't

| | OpenClaw / Hermes default | + This brain |
|---|---|---|
| Capacity | Last N chat turns (typically 20–50) | Unlimited — every fact the agent has ever seen |
| Persistence | Session-only (or simple `.md` files) | 4-tier CoALA model: working, episodic, semantic, procedural |
| Retrieval | Recency (last-message wins) | Hybrid vector + BM25 + keyword boost + temporal-proximity boost |
| Memory for the user | None (no user model) | Honcho-style `brain_user_model` block that accumulates over time |
| Memory for the work | Markdown files agents re-read every session | Wing/Room/Drawer 2D hierarchy (MemPalace-style): project → day → chunk |
| Forgetting | Linear (oldest message dropped) | FSRS-6 spaced repetition with self-tunable `w20` (per-deployment) |
| Consolidation | None | `reflect()` + `dreaming_cycle()` promote episodic → semantic |
| Conflict detection | None (new fact overwrites old) | mem0-style: near-duplicates marked `superseded_by` |
| Discovery | Grep through markdown | AAAK compression dialect scans the whole corpus in <500 tokens |
| Self-improvement | None | **Agent-driven skill pipeline**: agents stamp candidates with `brain_remember(kind="skill_candidate")` (no LLM) and promote them to agentskills.io SKILL.md themselves; `brain_skills_suggest` finds candidates by semantic query; `brain_optimize_fsrs` self-tunes the forgetting curve |
| Proactive | None | `brain_nudge` surfaces stale-but-important memories the agent is forgetting |
| Audit trail | None | Bi-temporal graph: `valid_from`/`valid_until` (world time) + `recorded_from`/`recorded_until` (when the brain knew) |

**TL;DR:** OpenClaw/Hermes remember the last 50 messages and call it memory. This brain makes the agent actually learn from the corpus it's seen.

It is designed for a practical loop:

1. Capture durable context from OpenClaw memory files, project docs, and direct `remember` calls.
2. Retrieve with hybrid vector + keyword + temporal-proximity search.
3. Consolidate episodic notes into semantic facts.
4. Sync useful memories back into agent context files.
5. Surface stale-but-important memories via proactive nudges.
6. Distill successful tasks into reusable agentskills.io SKILL.md manifests.

The project draws from mem0, Letta/MemGPT, Cognee, MemPalace, Graphiti, py-fsrs, Hermes Agent, and the CoALA memory taxonomy, but it keeps the runtime small: no general agent framework, no hosted database requirement, no secrets in client config. OpenClaw and Hermes are the agent runtimes that call into this brain; this repo is the memory layer they use.

## Core Capabilities

- **Four memory tiers:** working, episodic, semantic, and procedural.
- **Markdown-aware chunking:** recursive splitting around headers, paragraphs, sentences, and words.
- **Hybrid retrieval:** vector search + BM25-style keyword matching + Reciprocal Rank Fusion, with optional keyword-boost and temporal-proximity boost layers.
- **Entity and relationship memory:** lightweight graph storage for people, projects, files, and links between them, with bi-temporal edges (`valid_from`/`valid_until` for world time, `recorded_from`/`recorded_until` for when the brain learned it).
- **Verbatim recall:** exact source text retrieval for quotes, commands, and sensitive wording.
- **Memory health layers:** decay (Ebbinghaus), FSRS-6 spaced repetition with self-tunable `w20`, tier priors, rerank hooks, conflict detection (mem0-style), and injection scanning.
- **Wing/Room/Drawer 2D hierarchy (MemPalace-inspired):** people/projects × time × verbatim chunks, accessible via `brain_palace` MCP tool.
- **AAAK compression dialect:** compact one-line-per-chunk format for whole-corpus LLM scanning in <500 tokens, via `brain_index` MCP tool.
- **Honcho-style user modeling:** periodic distillation of user-related facts into a single `user` memory block, via `brain_user_model`.
- **Agent-driven skill pipeline:** agents stamp lightweight candidates with `brain_remember(kind="skill_candidate")` (no LLM call from the brain) and promote them to agentskills.io SKILL.md via `brain_skills_promote`. Supports `trust_level` ("full" skips injection scan; "standard" quarantines suspicious content) and `instructions_markdown` for rich SKILL.md bodies. The brain is pure storage + template — only the embedding model runs.
- **Skill auto-creation:** when an agent solves a new task, it can also distill it directly into an agentskills.io-compatible `SKILL.md` via `brain_skill_create`.
- **Skill suggestion:** `brain_skills_suggest` finds candidates by semantic query — "are there candidate skills about X?" — scoped to the procedural tier, filtered to unpromoted. No LLM.
- **Proactive memory nudges:** surface stale-but-important memories before they're forgotten, via `brain_nudge` MCP tool.
- **Local-first embeddings:** LM Studio works well locally; MiniMax, OpenAI, and sentence-transformers are also supported.
- **Watcher daemon:** polls markdown sources every five minutes by default and dedups unchanged content by hash.
- **MCP stdio server:** 66 tools for recall, remember, reflect, graph, blocks, dreaming, learning, quarantine, sync, wake-up, palace, index, nudge, skillgen, user modeling, agent-driven skill pipeline, export/import, and demo seeding.
- **Shell wrappers:** `scripts/duckbot-ask`, `scripts/brain-recall.sh`, `scripts/hermes-preflight.sh`, `scripts/hermes-postflight.sh`, and the one-command bootstraps `scripts/openclaw-bootstrap.sh` / `scripts/hermes-bootstrap.sh` for first-time setup.

> If you (or an agent) just want to install this without reading the README, see [INSTALL.md](INSTALL.md) for a single-page copy-paste recipe (prereqs → install → bootstrap → register MCP → cron).

## Quick Start

```bash
cd ~/Desktop/duckbot-rag-memory

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your embedding provider settings.

# One-command bootstrap for OpenClaw users: ingest every .md in
# ~/.openclaw/workspace into the brain and print the MCP registration step.
./scripts/openclaw-bootstrap.sh
# Same for Hermes Agent:
./scripts/hermes-bootstrap.sh

# Or do it manually:
python -m src.cli doctor
python -m src.cli ingest ~/.openclaw/workspace/memory
python -m src.cli query "What did we decide about cloud-only models?" -n 5

# See the full brain in one markdown file (for backup / migration):
python -m src.cli export --out-path data/brain_export.md

# Migrate an existing brain_export.md back into the brain:
python -m src.cli import data/brain_export.md
```

From any shell:

```bash
./scripts/duckbot-ask "What did we decide about cloud-only models?"
./scripts/duckbot-ask -f compact -n 3 "Duckets correction style"
./scripts/duckbot-ask -f snippet "BATMAN container restart recipe"
./scripts/brain-recall.sh "watcher restart steps"

# Manual / cron helpers (NOT auto-invoked by Hermes — the MemoryProvider
# plugin's on_session_start / on_session_end hooks cover session wiring):
./scripts/hermes-preflight.sh                 # one-shot wake_up block
./scripts/hermes-preflight.sh --query OpenClaw  # anchored on a topic
./scripts/hermes-postflight.sh                # nightly reflect pass
```

The wrappers load `.env` themselves and detect the local venv, so API keys do not need to be placed in MCP or shell history.

## Local Model Requirements

DuckBot does **not** ship model weights. If you want the default local LM Studio path to work, you must download these models yourself in LM Studio first. Without them, the default local setup will not run as documented:

- Required embeddings model: `text-embedding-embeddinggemma-300m`
- Required reranker model: `qwen3-reranker-0.6b`
- Optional consolidation model for `reflect()` and related distillation flows: `qwen2.5-7b-instruct` or another local LM Studio chat model

OpenClaw and Hermes are the agent runtimes that call into this repo. This repo is the memory layer they use; it does not provide its own chat model.

LM Studio is the preferred local path:

```bash
DUCKBOT_EMBEDDING=lmstudio
LMSTUDIO_URL=http://127.0.0.1:1234/v1
LMSTUDIO_API_KEY=lm-studio
LMSTUDIO_MODEL=text-embedding-embeddinggemma-300m
LMSTUDIO_RERANK_MODEL=qwen3-reranker-0.6b
DUCKBOT_RERANK=1
```

Other supported providers:

```bash
# Cloud fallback
DUCKBOT_EMBEDDING=minimax
MINIMAX_API_KEY=...

# OpenAI
DUCKBOT_EMBEDDING=openai
OPENAI_API_KEY=...

# Offline local model
DUCKBOT_EMBEDDING=local
```

If `DUCKBOT_EMBEDDING` is unset, the code auto-detects from available credentials and local services. Keep real keys only in `.env`; it is gitignored and protected by the secret-scan scripts.

If you do not install the reranker model, rerank stays available as a no-op fallback. If you do not install a consolidation chat model, `reflect()` falls back to regex-only extraction and still works, but with lower-quality semantic promotion.

## Watcher

Use the watcher for live-ish ingest instead of cron. It polls every five minutes by default, tracks content hashes, and skips no-op rewrites.

```bash
# One sync pass, useful for backfill or recovery.
python -m src.watcher once

# Foreground watcher.
python -m src.watcher run

# Detached launcher, recommended for macOS/Linux shells.
./scripts/start-watcher.sh

# Service integration.
./scripts/install-macos.sh   # launchd
./scripts/install-linux.sh   # systemd user service
pwsh scripts/install.ps1     # Windows Task Scheduler

# Status and stop.
python -m src.watcher status
python -m src.watcher stop
```

Default watch paths include OpenClaw memory files and selected project docs. Pass explicit paths to override:

```bash
python -m src.watcher run ~/.openclaw/workspace/memory ./AGENTS.md ./README.md
```

On macOS, polling is the default because `watchdog`/FSEvents has historically been unstable in the same process as ChromaDB and httpx. You can opt into native events with:

```bash
DUCKBOT_WATCH_USE_FSEVENTS=1 python -m src.watcher run
```

## MCP Integration

Run the memory server directly:

```bash
python -m src.mcp_server
```

Or register it with Hermes Agent:

```bash
hermes mcp add duckbot-memory \
  --command "$HOME/Desktop/duckbot-rag-memory/scripts/duckbot-memory-mcp.sh"
```

Windows:

```powershell
hermes mcp add duckbot-memory `
  --command "C:\Users\franz\Desktop\duckbot-rag-memory\scripts\duckbot-memory-mcp.bat"
```

Prefer the launcher scripts over passing env vars directly to MCP config. They load `.env` at process start, set unbuffered output, and choose the correct venv path per OS.

The current MCP server exposes 66 tools:

| Area | Tools |
| --- | --- |
| Core memory | `remember`, `recall`, `reflect`, `forget`, `stats`, `watch`, `doctor` |
| Retrieval maintenance | `recall_verbatim`, `search_verbatim`, `fsrs_review`, `decay_status`, `forget_by_query` |
| Enhanced brain | `brain_wake_up`, `brain_inflate`, `brain_sync`, `brain_recall`, `brain_remember`, `brain_reflect`, `brain_stats`, `brain_index`, `brain_nudge`, `brain_skill_create`, `brain_skills_list`, `brain_skills_suggest`, `brain_skills_promote`, `brain_user_model`, `brain_palace`, `brain_optimize_fsrs`, `brain_apply_fsrs_w20`, `brain_export`, `brain_import`, `brain_seed_demo` |
| Graph | `brain_graph_entity`, `brain_graph_relate`, `brain_graph_query`, `brain_graph_relationships`, `brain_graph_history`, `brain_graph_precursors`, `brain_graph_blind_spots` |
| Blocks | `brain_block_read`, `brain_block_write`, `brain_block_append`, `brain_block_delete`, `brain_block_list`, `brain_seed_blocks` |
| Safety | `brain_injection_scan`, `brain_quarantine_list`, `brain_quarantine_review` |
| Connectors | `dreaming_read`, `dreaming_cycle`, `learn`, `active_memory`, plus `brain_*` aliases |

See [docs/INTEGRATION.md](docs/INTEGRATION.md) for client-specific setup, Windows gotchas, and verification steps.

## Enhanced Brain

The enhanced brain closes the loop between retrieval and agent startup context.

```bash
# One-call session-start context load (Hermes pre-flight, OpenClaw init).
# Default output: markdown block ready to paste into an agent's context.
python -m src.cli wake-up

# Same data, JSON output for programmatic consumers.
python -m src.cli wake-up --json | jq '.memories'

# Sync useful memories into OpenClaw and Hermes context files.
python -m src.cli sync --target both

# Preview without writing.
python -m src.cli sync --dry-run

# Proactive nudge: what might the agent be forgetting?
python -m src.cli nudge --k 5 --min-importance 0.6

# Wing/Room/Drawer 2D view (MemPalace-inspired).
python -m src.cli palace                    # list all wings
python -m src.cli palace --wing openclaw    # walk one wing

# Self-tune the forgetting curve (FSRS w20).
python -m src.cli optimize-fsrs             # returns proposed w20 + sweep
python -m src.cli apply-fsrs-w20 0.42       # commit (persists via DUCKBOT_FSRS_W20)
```

`brain_wake_up` is the one-call session-start context load: top-k recent memories, active memory blocks, graph summary, and FSRS review queue. Designed for Hermes pre-flight + OpenClaw session-start hooks. `brain_inflate` recalls relevant memories and formats them as a markdown context block for an agent. `brain_sync` writes distilled context back to OpenClaw and Hermes memory files, respecting each platform's format and size limits. `brain_nudge` surfaces stale-but-important memories the agent might be forgetting. `brain_palace` exposes the MemPalace-inspired 2D hierarchy (person/project → time → verbatim chunk). `brain_optimize_fsrs` + `brain_apply_fsrs_w20` self-tune the FSRS-6 forgetting-curve exponent. `brain_skill_create` distills a successful task into an agentskills.io-compatible `SKILL.md`. `brain_user_model` aggregates user-related facts into a single Honcho-style user block.

## Agent-Driven Skill Pipeline

The brain never calls a generative LLM — only the embedding model runs. The agent authors skill content using its own LLM context; the brain is pure storage + template. This keeps the brain's VRAM footprint to just the embedder, regardless of how many candidates you accumulate.

The flow:

1. **Finish a task worth repeating** → stamp a lightweight candidate:
   ```
   brain_remember(
     text="Restarted BATMAN by running docker compose down && up",
     kind="skill_candidate",
     summary="BATMAN container restart",
     importance=0.8,
     trust_level="full",          # "full" (default) or "standard" (run injection scan)
   )
   ```
   The brain stores the chunk in the procedural tier with `metadata.kind="skill_candidate"`. No LLM call. Returns `chunk_id` immediately so you can promote it later.

2. **At a quiet moment** → review or search unpromoted candidates:
   ```
   brain_skills_list()                              # sorted by recency then importance
   brain_skills_suggest("docker container restart")  # semantic top-N by query
   ```

3. **Write the SKILL.md yourself** (you have the full context — what worked, what the user prefers, what to watch out for), then promote:
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
   )
   ```
   For richer SKILL.md bodies (headings, code blocks, tables), pass `instructions_markdown` instead of the flat `instructions` list:
   ```
   brain_skills_promote(
     chunk_id="mem_abc123",
     name="Restart BATMAN Container",
     description="Use this when BATMAN is offline",
     instructions_markdown="## Setup\n\nRun this first.\n\n## Usage\n\n```bash\ndocker compose up -d\n```",
     overwrite=True,
   )
   ```
   The brain writes `skills/<slug>/SKILL.md` via the existing `skillgen.write_skill` (pure template) and marks the candidate chunk as `promoted=True`.

**Standalone CLI** (no agent required):
```bash
python -m src.cli skills stamp "I learned to restart BATMAN via docker compose"
python -m src.cli skills list -k 10
python -m src.cli skills suggest "docker"
python -m src.cli skills promote <chunk_id> "Restart BATMAN" "When BATMAN is offline" "docker compose down" "docker compose up -d"
```

The same 12-tool core surface (which includes `brain_skills_list`, `brain_skills_suggest`, and `brain_skills_promote`) is exposed by every thin entry point: OpenClaw adapter, Hermes plugin, and the canonical MCP server. See [docs/PLUGIN_SURFACE.md](docs/PLUGIN_SURFACE.md) for the full comparison.

## Architecture

```text
Markdown sources
  -> watcher / CLI ingest / remember()
  -> chunk_markdown()
  -> tier classify + entity extraction + injection scan
  -> embeddings provider
  -> ChromaDB collections by tier
  -> hybrid query + RRF + optional recall adjustments
  -> CLI, MCP, OpenClaw, Hermes, Codex, Cursor
```

Storage lives under `data/` and is gitignored:

- `data/chroma/` for vector collections.
- `data/watcher_state.json` for file hashes and chunk IDs.
- `data/watcher.log` for daemon activity.
- `data/*.sqlite*` for graph, blocks, and quarantine state.

For the full design discussion, read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). For research lineage, read [docs/RESEARCH.md](docs/RESEARCH.md).

## Repository Map

```text
duckbot-rag-memory/
|-- src/
|   |-- chunk.py          # markdown-aware chunking
|   |-- tier.py           # CoALA tier classifier
|   |-- embeddings.py     # LM Studio, MiniMax, OpenAI, local providers
|   |-- store.py          # ChromaDB wrapper
|   |-- memory.py         # unified Memory facade
|   |-- query.py          # hybrid retrieval + RRF + keyword/temporal boost
|   |-- consolidate.py    # episodic -> semantic distillation
|   |-- mcp_server.py     # MCP stdio server (66 tools)
|   |-- watcher.py        # polling watcher daemon
|   |-- cli.py            # command-line interface
|   |-- dialect.py        # AAAK compression dialect (brain_index)
|   |-- palace.py         # Wing/Room/Drawer 2D index (brain_palace)
|   |-- skillgen.py       # agentskills.io SKILL.md generator (pure template)
|   |-- skill_pipeline.py # agent-driven candidate -> skill pipeline (no LLM)
|   |-- spellcheck.py     # common-typo fixer on ingest
|   |-- fsrs_optimizer.py # self-tune FSRS w20 from recall history
|   |-- blocks.py graph.py entities.py
|   |                      # blocks + temporal graph + entity extraction
|   |-- backends/         # chroma, lancedb, qdrant interfaces
|   |-- connectors/       # OpenClaw (legacy + aliases), Active Memory, dreaming, learn
|   |-- extensions/       # shared agent surface (12 tools) + generic JSON-RPC adapter for Claude Code/Cursor/Codex
|   `-- plugins/          # Hermes MemoryProvider plugin package
|-- tests/                # pytest suite, currently 849 tests
|-- benchmarks/           # golden retrieval evals
|-- extensions/           # native OpenClaw plugin (Node.js shim → Python MCP server)
|-- scripts/              # install, watcher, MCP launcher, query helpers,
|                         #   hermes-preflight.sh, hermes-postflight.sh
|-- skills/               # OpenClaw skill manifests and plugins
|-- docs/                 # architecture, integration, research
|-- data/                 # gitignored runtime state
|-- .env.example          # local config template
|-- AGENTS.md             # instructions for coding agents
|-- CHANGELOG.md          # release history
|-- INSTALL.md           # one-page install recipe for OpenClaw/Hermes agents
|-- CONTRIBUTING.md       # contribution guide
|-- SECURITY.md           # disclosure and hardening notes
`-- README.md
```

## CLI Reference

```bash
python -m src.cli ingest <path> [<path> ...]
python -m src.cli query "question" -n 5
python -m src.cli stats
python -m src.cli eval benchmarks/golden.jsonl
python -m src.cli consolidate 7
python -m src.cli compact
python -m src.cli dashboard --json
python -m src.cli sync --target openclaw|hermes|both
python -m src.cli skills stamp "I learned to restart BATMAN"
python -m src.cli skills list
python -m src.cli skills promote <chunk_id> <name> <description> <instr1> [instr2 ...]
python -m src.cli skills suggest "docker container restart"
python -m src.cli doctor
```

`reset` exists for local development, but it wipes collections and requires `--yes`.

## Testing

```bash
pytest -v
pytest tests/test_memory.py -v
pytest -k "watcher" -v
pytest --collect-only -q
bash scripts/secret-scan.sh
```

Current local check: **849 tests passing**. The suite is exercised in CI on every push.

Eval trend detection (`compute_trend`) reads `data/eval_history.jsonl` and reports recent-vs-prior deltas on `mean_recall_at_5`, `mean_mrr`, and `p95_latency`. `python -m src.cli eval benchmarks/golden.jsonl` prints the trend alongside the summary.

## Cross-Platform Notes

| Task | macOS/Linux | Windows |
| --- | --- | --- |
| Python | `./.venv/bin/python` | `.venv\Scripts\python.exe` |
| Install | `./scripts/install.sh` | `pwsh scripts/install.ps1` |
| Start watcher | `./scripts/start-watcher.sh` | `pwsh scripts/start-watcher.ps1` |
| MCP launcher | `scripts/duckbot-memory-mcp.sh` | `scripts\duckbot-memory-mcp.bat` |
| Secret scan | `bash scripts/secret-scan.sh` | `pwsh scripts/secret-scan.ps1` |

Python 3.12+ is recommended. The Xcode-shipped Python 3.9 on macOS has caused ChromaDB crashes in this project.

## Project Rules

- Keep ingest idempotent: same content should produce stable chunk IDs.
- Keep storage tiered: working memory can age out; procedural memory should not.
- Keep secrets out of source, MCP config, logs, and docs.
- Prefer additive changes and migrations over destructive rewrites.
- Update docs and changelog when behavior changes.
- Run the secret scan before committing.

## License

MIT. See [LICENSE](LICENSE).
