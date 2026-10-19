# DuckBot RAG + Memory System

Persistent, searchable memory that dramatically expands the limited default memory of Hermes Agent (the primary runtime today) and OpenClaw (legacy, still synced). Drop-in replacement for the flat chat-history buffers both agents ship with — adds a 4-tier CoALA memory model, hybrid retrieval, entity graph, verbatim recall, FSRS-6 spaced repetition, dreaming consolidation, and a Wing/Room/Drawer 2D hierarchy on top.

> **What's new — 2026-10-19: Hy-Memory is the live consolidation writer.** The writer path below cleared its 2026-10-16 evaluation week (nightly shadow A/B over real memory files: Hy-Memory 107 facts at a 0.243 duplicate rate vs the regex incumbent's 40 at 0.65) and was promoted — a nightly pass (`scripts/nightly_hy_pass.py`, Hermes cron, no-agent) now runs the Hy-Memory load-on-dispatch writer against the live bank. Chat model and embedder stay loaded; the writer is unloaded after each run. See [The Hy-Memory Writer Path](#the-hy-memory-writer-path) and the [deployed jobs table](#deployed-jobs-reference-mac).
>
> **2026-10-16: the Hy-Memory writer path.** Consolidation is no longer regex-only. A dedicated writer model — Tencent's Hy-Memory-30B-A3B, run locally as a 4-bit MLX build in LM Studio — extracts memory statements behind a SystemOne/D1 quality gate, with mem0-style update semantics (add / update / retire-for-review). It was A/B-validated against the regex incumbent in a nightly shadow run during the evaluation week, then promoted to the live nightly consolidation on 2026-10-19.

[![Status](https://img.shields.io/badge/latest_changelog-0.15.3-yellow)]()
[![MCP](https://img.shields.io/badge/MCP_server-0.15.3-green)]()
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
| Consolidation | Agent-driven | `reflect()` + `dreaming_cycle()` promote episodic → semantic with agent-supplied facts or regex fallback; an optional dedicated writer model (Hy-Memory, local MLX) behind a D1 gate when the batch job opts in |
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
- **MCP stdio server:** 73 tools for recall, remember, reflect, graph, blocks, dreaming, learning, quarantine, sync, wake-up, palace, index, nudge, skillgen, user modeling, agent-driven skill pipeline, export/import, and demo seeding.
- **Shell wrappers:** `scripts/duckbot-ask`, `scripts/brain-recall.sh`, `scripts/hermes-preflight.sh`, `scripts/hermes-postflight.sh`, and the one-command bootstraps `scripts/openclaw-bootstrap.sh` / `scripts/hermes-bootstrap.sh` for first-time setup.

> If you (or an agent) just want to install this without reading the README, see [INSTALL.md](INSTALL.md) for a single-page copy-paste recipe (prereqs → install → bootstrap → register MCP → cron).

## The Hy-Memory Writer Path

Historically, consolidation extracted facts two ways: the agent supplied them, or `reflect()` fell back to lightweight regex heuristics (`src/consolidate.py`). Regex is free and predictable, but it misses most of what a conversation actually established. The Hy-Memory path adds a third writer — a model built for exactly this job — the nightly consolidation pass is that opt-in: `scripts/nightly_hy_pass.py` runs `reflect()` with the Hy extraction callback (fail-open to regex per chunk), and it has been the live writer since 2026-10-19.

**The writer.** Tencent's [Hy-Memory-30B-A3B](https://huggingface.co/tencent/Hy-Memory-30B-A3B) (Apache-2.0; 30B total / ~3B active MoE, `hy_v3`, 262K context) is a dedicated memory-extraction model. On Tencent's cross-harness evaluation (Mem0 / MemOS / Hy-Memory / EverMemOS) it outscored Claude Opus 5 and Qwen3.6-35B-A3B as a drop-in memory writer. We run it locally as a **4-bit MLX build in LM Studio** — [`Duckets/Hy-Memory-30B-A3B-MLX-4bit`](https://huggingface.co/Duckets/Hy-Memory-30B-A3B-MLX-4bit) (affine, group 64, ~17 GB, identifier `hy-memory-30b-a3b-mlx`). The adapter is `src/hy_memory.py`: explicit `model=` only, no environment defaults, silent empty-list fallback — it can never load a model by accident. The writer is **load-on-dispatch**: jobs load it, use it, and unload it; it is never left resident.

**The gate.** Not every chunk deserves the writer's time or a slot in the bank. `src/gate.py` asks the SystemOne decision layer (D1, `d1-3b` in LM Studio, via `POST /v1/systemone`) one 4-way question per chunk — route to bank `chat`, `working`, or `code`, or `none`. A chunk is skipped only when `none` wins at ≥ 0.60 confidence. The gate **fails open**: any error stores the chunk ungated rather than dropping memory. Calibration on real chunks: bio → chat 0.92, junk → none 0.84, project → working 0.97, preferences → chat 0.76.

**Update semantics.** Extraction alone creates duplicates and contradictions, so the writer also plans *operations* against existing memories (`plan_memory_updates` / `parse_memory_ops` in `src/hy_memory.py`): `add` for new facts, `update` superseding a stale statement, and `retire` — which is **review-only and never auto-applied**, because early runs showed the model occasionally retiring an unrelated memory.

**The batch job.** `scripts/consolidate_job.py` is the cutover-ready entrypoint: `--writer regex|hy-memory` (default regex = today's incumbent behavior), gate on by default, the writer role pinned as SystemOne's `memory_writer`. It is **dry-run by default**; `--apply` is wired (the hy-memory writer runs with Phase A reconciliation). For the live bank, prefer `scripts/nightly_hy_pass.py` — the deployed nightly consolidation pass (see the [deployed jobs table](#deployed-jobs-reference-mac)).

**The shadow A/B.** `scripts/shadow_writer_ab.py` runs both writers over the same chunks and writes a comparison report to `data/shadow/` — it imports no store code and *cannot* write to the bank. Its corpus is the most recent memory files across **both** agent homes (`~/.hermes/memories` and `~/.openclaw/workspace/memory`), so it always evaluates on whichever agent is actually in use. It ran nightly during the 2026-10-16 evaluation week; the evaluation is complete — Hy-Memory was promoted to the live writer on 2026-10-19. The script stays for future writer comparisons. First head-to-head on real consolidated memory files (2026-10-16): incumbent 40 facts at a 0.65 duplicate rate; Hy-Memory 107 facts (81 unique) at 0.243.

**End-to-end proof.** `scripts/e2e_pipeline_test.py` exercises the whole path against a throwaway bank — chunk → gate → extract → update ops → apply → store + recall probes — and currently passes every stage on the MLX writer. Unit coverage lives in `tests/test_hy_memory.py`, `tests/test_gate.py`, and `tests/test_hy_memory_updates.py`.

## Quick Start

**Quick start** — the one command to remember:

| | Command |
|---|---|
| macOS/Linux | `duck-memory query "what did we decide?"` |
| Windows | `duck-memory.bat query "what did we decide?"` |

**One-command setup** (recommended — picks your OS automatically):

| OS | Setup | Update | Demo | Start watcher |
|---|---|---|---|---|
| macOS | Double-click `scripts/setup.command` | Double-click `scripts/update.command` | Double-click `scripts/demo.command` | Double-click `scripts/start.command` |
| Linux | `./scripts/setup.sh` | `./scripts/update.sh` | `./scripts/demo.sh` | `./scripts/start.sh` |
| Windows | Double-click `scripts\setup.bat` | Double-click `scripts\update.bat` | Double-click `scripts\demo.bat` | Double-click `scripts\start.bat` |

The setup script creates the venv, installs deps, seeds the demo corpus, runs a sample query, and prints next steps. Update pulls the latest from GitHub and upgrades deps.

**What you get in 30 seconds:**

```bash
# Query the brain — use duck-memory from anywhere on PATH
duck-memory query "How do I restart the BATMAN container?"
duck-memory query -n 5 "DuckBot design constraints"

# Or from the repo directory
./duck-memory query "what did we decide?"
./scripts/duckbot-ask -f snippet "PRL pool wallet steps"

# Start the watcher daemon (recommended — polls every 5 min)
./scripts/start.sh          # macOS / Linux
# Windows: double-click scripts\start.bat

# Run the demo again
./scripts/demo.sh            # macOS / Linux
# Windows: double-click scripts\demo.bat

# Bootstrap with OpenClaw or Hermes Agent
./scripts/openclaw-bootstrap.sh
./scripts/hermes-bootstrap.sh

# Query from any shell (loads .env, cross-platform)
./scripts/duckbot-ask "What did we decide about cloud-only models?"
./scripts/duckbot-ask -f compact -n 3 "Duckets correction style"
```

# See the full brain in one markdown file (for backup / migration):
python -m src.cli export --out-path data/brain_export.md

# Migrate an existing brain_export.md back into the brain:
python -m src.cli import data/brain_export.md
```

From any shell:

```bash
# Update to the latest version
./scripts/update.sh                              # macOS / Linux
# Windows: double-click scripts\update.bat

# Agents use the CLI directly (returns JSON):
python -m src.cli update                        # pull + upgrade deps + doctor
python -m src.cli update --dry-run             # check if updates available
python -m src.cli update --no-deps --no-doctor  # pull only, skip upgrade/checks

./scripts/duckbot-ask "What did we decide about cloud-only models?"
./scripts/duckbot-ask -f compact -n 3 "Duckets correction style"
./scripts/duckbot-ask -f snippet "BATMAN container restart recipe"
./scripts/brain-recall.sh "watcher restart steps"

# Manual / cron helpers (the MemoryProvider plugin's on_session_start /
# on_session_end hooks cover session wiring where it can load — on the
# reference Mac the plugin self-disables because Hermes' isolated backend
# Python has no chromadb, so the MCP brain_* tools + the nightly jobs do):
./scripts/hermes-preflight.sh                 # one-shot wake_up block
./scripts/hermes-preflight.sh --query OpenClaw  # anchored on a topic
./scripts/hermes-postflight.sh                # nightly reflect pass
```

The wrappers load `.env` themselves and detect the local venv, so API keys do not need to be placed in MCP or shell history.

## Local Model Requirements

DuckBot does **not** ship model weights.

### Deployed configuration (the reference Mac — read this first)

The live system uses **LM Studio** for embeddings, and the bank is built at that dimension — **never change the dimension source on an existing bank, or every stored vector is invalidated**:

```bash
DUCKBOT_EMBEDDING=lmstudio
LMSTUDIO_MODEL=text-embedding-nomic-embed-text-v1.5   # 768-dim
LMSTUDIO_EMBED_DIM=768
```

| Role | Model | Notes |
|---|---|---|
| Embeddings | LM Studio `text-embedding-nomic-embed-text-v1.5` (768-dim) | `DUCKBOT_EMBEDDING=lmstudio`; the nightly jobs self-load it |
| Reranker | Qwen3 `qwen3-reranker-0.6b` (LM Studio) | decay scoring; no-op fallback if absent |
| Decision gate | `d1-3b` via SystemOne (`POST /v1/systemone`) | kept warm by SystemOne; fails open |
| Memory writer | `hy-memory-30b-a3b-mlx` — Hy-Memory 4-bit MLX in LM Studio | **live nightly consolidation writer**; load-on-dispatch; see [The Hy-Memory Writer Path](#the-hy-memory-writer-path) |
| Chat model | `qwen3.8-27b-splash` (LM Studio) | Hermes default — standing model rule; never punted by the brain jobs |

### Alternative embedding providers

If you build a *new* bank on a different provider, LM Studio embeddings work well (the live bank runs `text-embedding-nomic-embed-text-v1.5`, 768-dim; `DUCKBOT_EMBEDDING=lmstudio`), as do the cloud providers below. Fact extraction remains available agent-side regardless of provider: pass facts via `brain_remember(facts=[...])` or `reflect(extract_callback=...)`; `reflect()` uses the regex heuristics when no facts are supplied.

OpenClaw and Hermes are the agent runtimes that call into this repo. This repo is the memory layer they use; it does not provide its own chat model.

LM Studio is the preferred local path:

```bash
DUCKBOT_EMBEDDING=lmstudio
LMSTUDIO_URL=http://127.0.0.1:1234/v1
LMSTUDIO_KEY=<your LM Studio key>
LMSTUDIO_MODEL=text-embedding-nomic-embed-text-v1.5
LMSTUDIO_EMBED_DIM=768
LMSTUDIO_RERANK_MODEL=qwen3-reranker-0.6b
DUCKBOT_RERANK=1
# No special flag needed for regex-only reflect(); it is the default
# when no agent facts are supplied.
```

Other supported providers:

```bash
# NVIDIA NIM (OpenAI-compatible cloud embeddings)
DUCKBOT_EMBEDDING=nvidia
NVIDIA_API_KEY=...
NVIDIA_EMBED_MODEL=nvidia/nemotron-3-embed-1b
```

```bash
# Cloud fallback
DUCKBOT_EMBEDDING=minimax
MINIMAX_API_KEY=...

# OpenAI
DUCKBOT_EMBEDDING=openai
OPENAI_API_KEY=...

# Offline local model (embeddings via sentence-transformers; no LM Studio needed for embeddings)
DUCKBOT_EMBEDDING=local
```

If `DUCKBOT_EMBEDDING` is unset, the code auto-detects from available credentials and local services. Keep real keys only in `.env`; it is gitignored and protected by the secret-scan scripts.

If you do not install the reranker model, rerank stays available as a no-op fallback. Fact extraction is agent-driven by default (pass facts to `brain_remember(facts=[...])` or `reflect(extract_callback=...)`); the optional Hy-Memory writer path above is the one exception, and only for batch jobs that explicitly opt in.

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

### Deployed jobs (reference Mac)

| Job | Schedule | What it does |
|---|---|---|
| Watcher (`com.duckbot.memory-watcher`, launchd) | continuous | Ingests the OpenClaw workspace memory files (`~/.openclaw/workspace/memory` + workspace root files); `brain_sync` keeps the Hermes and OpenClaw context files in sync, so Hermes-written memory flows in too |
| Hy-Memory consolidation (Hermes cron, no-agent) | nightly 03:00 | Episodic → semantic consolidation with the Hy-Memory writer — load-on-dispatch (`hy-memory-30b-a3b-mlx`, unloaded after), fail-open to regex per chunk; chat model + embedder stay loaded |
| Brain decay (no-agent) | daily 03:00 | FSRS retention-floor pruning, scored by `qwen3-reranker-0.6b` |
| FSRS optimize (no-agent, no LLM) | weekly, Sun 04:00 | Self-tunes the forgetting curve |
| Brain-sync → Hermes files (no-agent) | hourly | `brain_sync` writes SOUL.md / MEMORY.md / USER.md |
| Service guardian RAG-watch (no-agent) | 08:00 / 14:00 / 20:00 | Keeps the MCP server + watcher alive |
| ~~Brain Reflect (agent job)~~ | paused — superseded | Was weekly episodic → semantic, pinned to LFM2.5-24B-A2B (not served on the reference Mac); superseded by the Hy-Memory nightly consolidation |
| ~~Hy-Memory shadow A/B~~ | evaluation complete (2026-10-16 week) | Read-only writer comparison over both agent homes; Hy-Memory promoted to the live writer — evaluation closed |

### Hermes MemoryProvider plugin — install path matters

The bootstrap (`scripts/hermes-bootstrap.sh`) installs the brain plugin to the **canonical top-level location** `~/.hermes/plugins/duckbot-brain/` (dash name) and sets `memory.provider: duckbot-brain` in `~/.hermes/config.yaml`. This exact path is load-bearing: Hermes' memory-provider resolver (`plugins.memory.find_provider_dir`) only looks for a top-level `plugins/<provider-name>/` directory. The older nested layout (`plugins/memory/duckbot_brain/`) is found by the general plugin scanner — `hermes plugins list` shows it enabled — but **not** by the resolver, which produces a misleading "memory provider 'duckbot-brain' is configured but not installed" banner. If you ever see that banner while the plugin lists as enabled, check the resolver path first, not the config and not the catalog.

## MCP Integration

Run the memory server directly:

```bash
python -m src.mcp_server
```

Or register it with Hermes Agent:

```bash
hermes mcp add duckbot-memory \
  --command "$HOME/duckbot-rag-memory/scripts/duckbot-memory-mcp.sh"
```

Windows:

```powershell
hermes mcp add duckbot-memory `
  --command "C:\Users\franz\Desktop\duckbot-rag-memory\scripts\duckbot-memory-mcp.bat"
```

Prefer the launcher scripts over passing env vars directly to MCP config. They load `.env` at process start, set unbuffered output, and choose the correct venv path per OS.

The current MCP server exposes 73 tools:

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

The same 73-tool core surface (which includes `brain_skills_list`, `brain_skills_suggest`, and `brain_skills_promote`) is exposed by every thin entry point: OpenClaw adapter, Hermes plugin, and the canonical MCP server. See [docs/PLUGIN_SURFACE.md](docs/PLUGIN_SURFACE.md) for the full comparison.

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
|   |-- consolidate.py    # episodic -> semantic distillation (regex incumbent)
|   |-- hy_memory.py      # Hy-Memory writer adapter + update-op planner (explicit model only)
|   |-- gate.py           # SystemOne/D1 chunk gate: chat|working|code|none, fail-open
|   |-- mcp_server.py     # MCP stdio server (73 tools)
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
|   |-- extensions/       # shared agent surface (73 tools) + generic JSON-RPC adapter for Claude Code/Cursor/Codex
|   `-- plugins/          # Hermes MemoryProvider plugin package
|-- tests/                # pytest suite, currently 849 tests
|-- benchmarks/           # golden retrieval evals
|-- extensions/           # native OpenClaw plugin (Node.js shim → Python MCP server)
|-- scripts/              # install, watcher, MCP launcher, query helpers,
|                         #   hermes-preflight.sh, hermes-postflight.sh,
|                         #   shadow_writer_ab.py (read-only writer A/B),
|                         #   consolidate_job.py (cutover-ready batch job),
|                         #   e2e_pipeline_test.py (full-path proof)
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

The suite stood at 849 tests at v0.15.3 and is exercised in CI on every push. The Hy-Memory path adds `tests/test_hy_memory.py`, `tests/test_gate.py`, and `tests/test_hy_memory_updates.py`, plus a full-pipeline end-to-end check that runs against a throwaway bank (never the live one):

```bash
DUCKBOT_EMBEDDING=local .venv/bin/python scripts/e2e_pipeline_test.py
# chunk -> gate -> extract -> update ops -> apply -> store + recall probes
```

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
