# DuckBot RAG + Memory System

Persistent, searchable memory for DuckBot, OpenClaw, Hermes Agent, and any MCP client.

[![Status](https://img.shields.io/badge/latest_changelog-0.11.5-yellow)]()
[![MCP](https://img.shields.io/badge/MCP_server-0.11.7-green)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![CI](https://github.com/Franzferdinan51/duckbot-rag-memory/actions/workflows/ci.yml/badge.svg)]()

## What This Is

DuckBot memory is a focused RAG and long-term memory layer for personal agent workflows. It ingests markdown, classifies it into memory tiers, embeds it, stores it in local vector collections, and exposes recall/write tools through CLI wrappers and MCP.

It is designed for a practical loop:

1. Capture durable context from OpenClaw memory files, project docs, and direct `remember` calls.
2. Retrieve with hybrid vector + keyword search.
3. Consolidate episodic notes into semantic facts.
4. Sync useful memories back into agent context files.

The project draws from mem0, Letta/MemGPT, Cognee, Hermes Agent, and the CoALA memory taxonomy, but it keeps the runtime small: no general agent framework, no hosted database requirement, no secrets in client config.

## Core Capabilities

- **Four memory tiers:** working, episodic, semantic, and procedural.
- **Markdown-aware chunking:** recursive splitting around headers, paragraphs, sentences, and words.
- **Hybrid retrieval:** vector search + BM25-style keyword matching + Reciprocal Rank Fusion.
- **Entity and relationship memory:** lightweight graph storage for people, projects, files, and links between them.
- **Verbatim recall:** exact source text retrieval for quotes, commands, and sensitive wording.
- **Memory health layers:** decay, FSRS-style review scheduling, tier priors, rerank hooks, and injection scanning.
- **Local-first embeddings:** LM Studio works well locally; MiniMax, OpenAI, and sentence-transformers are also supported.
- **Watcher daemon:** polls markdown sources every five minutes by default and dedups unchanged content by hash.
- **MCP stdio server:** 45 tools for recall, remember, reflect, graph, blocks, dreaming, learning, quarantine, and sync.
- **Shell wrappers:** `scripts/duckbot-ask` and `scripts/brain-recall.sh` for use from any terminal or cron job.

## Quick Start

```bash
cd ~/Desktop/duckbot-rag-memory

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env with your embedding provider settings.

python -m src.cli doctor
python -m src.cli ingest ~/.openclaw/workspace/memory
python -m src.cli query "What did we decide about cloud-only models?" -n 5
```

From any shell:

```bash
./scripts/duckbot-ask "What did we decide about cloud-only models?"
./scripts/duckbot-ask -f compact -n 3 "Duckets correction style"
./scripts/duckbot-ask -f snippet "BATMAN container restart recipe"
./scripts/brain-recall.sh "watcher restart steps"
```

The wrappers load `.env` themselves and detect the local venv, so API keys do not need to be placed in MCP or shell history.

## Recommended Embedding Setup

LM Studio is the preferred local path:

```bash
DUCKBOT_EMBEDDING=lmstudio
LMSTUDIO_URL=http://127.0.0.1:1234/v1
LMSTUDIO_API_KEY=lm-studio
LMSTUDIO_MODEL=text-embedding-nomic-embed-text-v1.5
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

The current MCP server exposes 45 tools:

| Area | Tools |
| --- | --- |
| Core memory | `remember`, `recall`, `reflect`, `forget`, `stats`, `watch`, `doctor` |
| Retrieval maintenance | `recall_verbatim`, `search_verbatim`, `fsrs_review`, `decay_status`, `forget_by_query` |
| Enhanced brain | `brain_inflate`, `brain_sync`, `brain_recall`, `brain_remember`, `brain_reflect`, `brain_stats` |
| Graph | `brain_graph_entity`, `brain_graph_relate`, `brain_graph_query`, `brain_graph_relationships`, `brain_graph_history` |
| Blocks | `brain_block_read`, `brain_block_write`, `brain_block_append`, `brain_block_delete`, `brain_block_list`, `brain_seed_blocks` |
| Safety | `brain_injection_scan`, `brain_quarantine_list`, `brain_quarantine_review` |
| Connectors | `dreaming_read`, `dreaming_cycle`, `learn`, `active_memory`, plus `brain_*` aliases |

See [docs/INTEGRATION.md](docs/INTEGRATION.md) for client-specific setup, Windows gotchas, and verification steps.

## Enhanced Brain

The enhanced brain closes the loop between retrieval and agent startup context.

```bash
# Sync useful memories into OpenClaw and Hermes context files.
python -m src.cli sync --target both

# Preview without writing.
python -m src.cli sync --dry-run
```

`brain_inflate` recalls relevant memories and formats them as a markdown context block for an agent. `brain_sync` writes distilled context back to OpenClaw and Hermes memory files, respecting each platform's format and size limits.

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
|   |-- query.py          # hybrid retrieval and RRF
|   |-- consolidate.py    # episodic -> semantic distillation
|   |-- mcp_server.py     # MCP stdio server
|   |-- watcher.py        # polling watcher daemon
|   |-- cli.py            # command-line interface
|   |-- backends/         # chroma, lancedb, qdrant interfaces
|   `-- connectors/       # OpenClaw, Active Memory, dreaming, learn
|-- tests/                # pytest suite, currently 560 tests collected
|-- benchmarks/           # golden retrieval evals
|-- scripts/              # install, watcher, MCP launcher, query helpers
|-- skills/               # OpenClaw skill manifests and plugins
|-- docs/                 # architecture, integration, research
|-- data/                 # gitignored runtime state
|-- .env.example          # local config template
|-- AGENTS.md             # instructions for coding agents
|-- CHANGELOG.md          # release history
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

Current local collection check: **560 tests collected**. The latest changelog entry records **529 passing** for the 0.11.5 audit fixes; the local suite has continued to grow since then.

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
