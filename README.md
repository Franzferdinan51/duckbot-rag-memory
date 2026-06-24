# 🦆 DuckBot Memory System

> Persistent, searchable, and self-curing memory for OpenClaw + Hermes Agent. Auto-updates in real time.

[![Status](https://img.shields.io/badge/status-v0.2-yellow)]() [![License](https://img.shields.io/badge/license-MIT-blue)]() [![Open Source](https://img.shields.io/badge/open--source-everything-green)]()

## What this is

A **RAG pipeline + memory layer + auto-updater** built specifically for personal AI agent usage. Combines:

- **RAG core** — markdown-aware chunking, vector + BM25 hybrid retrieval, Reciprocal Rank Fusion
- **CoALA 4-tier** — working / episodic / semantic / procedural memory taxonomy
- **Entity memory** — people, places, orgs, products with relationships (Cognee-inspired)
- **Auto-updater** — file-watcher daemon syncs new/changed markdown in real time (mem0 hook pattern)
- **Sleep-time consolidation** — `reflect()` pass that promotes episodic → semantic
- **Importance scoring** — recall bumps importance; old + unused decays naturally
- **Pluggable embeddings** — LM Studio (primary), MiniMax (fallback), OpenAI, sentence-transformers
- **MCP server** — `remember` / `recall` / `reflect` / `forget` / `stats` tools for any MCP client

Inspired by (and pulling from):
- **mem0** (mem0ai/mem0) — hook-based auto-capture, `add`/`update`/`search` API
- **Letta / MemGPT** (letta-ai/letta) — tiered memory, self-editing blocks, sleep-time agents
- **Cognee** (topoteretes/cognee) — ECL pipeline (Extract → Cognify → Load), knowledge graph
- **Hermes Agent** (NousResearch/hermes-agent) — FTS5 session search, LLM summarization
- **CoALA framework** (Princeton 2023) — 4-tier memory taxonomy
- **LangChain** RecursiveCharacterTextSplitter, **LlamaIndex** SentenceSplitter

## Quick start

```bash
cd ~/Desktop/duckbot-rag-memory
./.venv/bin/python -m src.cli doctor          # verify everything works
./.venv/bin/python -m src.cli watch once      # one-shot full sync (~5 min for 150 files)
./.venv/bin/python -m src.watcher daemon      # run auto-updater in background
./.venv/bin/python -m src.cli query "What did we decide about cloud-only models?" -n 5
```

Or programmatically:

```python
from src.memory import Memory
mem = Memory()
await mem.remember("Today we installed cua-driver v0.6.2.")
results, stats = await mem.recall("cua-driver", k=3)
await mem.reflect()  # sleep-time consolidation
```

## Architecture (one page)

```
┌───────────────────────────────────────────────────────────────┐
│                      file watcher (real time)                  │
│  /memory/*.md, AGENTS.md, SOUL.md, project docs               │
│  ↓ on change                                                   │
│  ┌────────────────────────────────────────────────────────┐   │
│  │            Memory.remember(text)                        │   │
│  │  chunk → tier classify → entity extract → importance   │   │
│  │           → embed (LM Studio / MiniMax / OpenAI)        │   │
│  │           → upsert into ChromaDB (idempotent)           │   │
│  └────────────────────────────────────────────────────────┘   │
│                              ↓                                 │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ ChromaDB (one collection per tier)                     │   │
│  │   duckbot_working   (capped, recent sessions)          │   │
│  │   duckbot_episodic  (daily logs, dated)                │   │
│  │   duckbot_semantic  (distilled facts, entities)        │   │
│  │   duckbot_procedural(rules, norms, AGENTS/SOUL)        │   │
│  └────────────────────────────────────────────────────────┘   │
│                              ↓                                 │
│  ┌────────────────────────────────────────────────────────┐   │
│  │ Memory.recall(query)                                    │   │
│  │  embed query → vector search + BM25 → RRF fuse         │   │
│  │  → bump importance of returned chunks                  │   │
│  │  → spread activation to related chunks                 │   │
│  └────────────────────────────────────────────────────────┘   │
│                              ↓                                 │
│  OpenClaw agents / Claude Code / Cursor / Codex via MCP        │
│  (tools: remember, recall, reflect, forget, stats, watch)      │
└───────────────────────────────────────────────────────────────┘
```

## Auto-update (no cron)

Per Duckets (2026-06-23): **no automatic cron** — memory updates in real time.

```bash
# 1. One-shot full sync (good for cold start or recovery)
./.venv/bin/python -m src.watcher once

# 2. Run as a long-lived process (foreground; survives exec-tool sessions)
nohup ./.venv/bin/python -m src.watcher run </dev/null >data/watcher.log 2>&1 &

# 3. Or wire it into launchd (auto-restart on crash + on boot)
cp scripts/com.duckbot.memory-watcher.plist ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/com.duckbot.memory-watcher.plist

# 4. Or trigger via MCP from any agent
call_tool("watch", {"daemon": true})
```

Management:

```bash
./.venv/bin/python -m src.watcher status   # check if running
./.venv/bin/python -m src.watcher stop     # SIGTERM the daemon
```

The watcher (uses `watchdog` for FSEvents/inotify, falls back to polling) watches:
- `~/.openclaw/workspace/memory/` (all daily logs)
- `~/.openclaw/workspace/MEMORY.md`, `AGENTS.md`, `SOUL.md`, `IDENTITY.md`
- `~/Desktop/ai-Py-boy-emulation-main/` (project docs)
- `~/Desktop/Newest Desktop Control/` (project docs)

Override with extra paths: `./.venv/bin/python -m src.watcher run <path1> <path2>`.

### Why polling, not watchdog

On macOS, `watchdog` (FSEvents) combined with `chromadb` + `httpx` in the same
process segfaults reliably. The polling handler (default since 2026-06-23) is
rock-solid and trades a 2-second poll for that stability. To opt back into
FSEvents: `DUCKBOT_WATCH_USE_FSEVENTS=1 ./.venv/bin/python -m src.watcher run`.

### Why `run` not `daemon`

`watcher daemon` uses an internal double-fork that gets killed by SIGHUP from the
parent shell on macOS (the orphaned grandchild inherits a defunct controlling tty).
The working pattern is `subprocess.Popen(start_new_session=True)` from a Python
helper script (see `scripts/start-watcher.sh`), or `launchd` which provides a
proper background process tree. The `daemon` subcommand is kept for compatibility
but not the recommended path on macOS.

### Why Python 3.12, not 3.9

The Xcode-shipped Python 3.9.6 + chromadb 1.5.9 + arm64 segfaults on `coll.count()`
and `coll.upsert()`. The fix is to recreate the venv with the homebrew Python 3.12
or 3.13:

```bash
cd ~/Desktop/duckbot-rag-memory
/opt/homebrew/bin/python3.12 -m venv .venv
# If the symlinks still point to Python 3.9, fix them:
cd .venv/bin && rm -f python python3 python3.9 && ln -s python3.12 python3 && ln -s python3.12 python
cd ../.. && ./.venv/bin/python -m pip install -r requirements.txt
```

## Embedding providers

Per Duckets (2026-06-23): **LM Studio primary, MiniMax fallback.**

Set in `.env`:
```bash
DUCKBOT_EMBEDDING=lmstudio  # primary
LMSTUDIO_URL=http://127.0.0.1:1234/v1
LMSTUDIO_KEY=sk-lm-xxx:yyy
LMSTUDIO_MODEL=text-embedding-embeddinggemma-300m

# Fallback (cloud, paid, high quality)
MINIMAX_API_KEY=sk-cp-xxx
MINIMAX_BASE_URL=https://api.minimax.io/v1
```

Auto-detect chain: `DUCKBOT_EMBEDDING` env > LM Studio reachable > MiniMax key > OpenAI key > sentence-transformers.

## MCP server (for any agent)

Exposes the memory API to MCP clients. Run as stdio server:

```bash
./.venv/bin/python -m src.mcp_server
```

Tools: `remember`, `recall`, `reflect`, `forget`, `stats`, `watch`, `doctor`.

Wire it into OpenClaw, Claude Code, Cursor, Codex via their respective MCP configs. The server returns JSON-RPC 2.0 responses on stdout.

## Files

```
duckbot-rag-memory/
├── src/
│   ├── chunk.py          # markdown-aware recursive chunker (512 tok, 15% overlap)
│   ├── tier.py           # CoALA 4-tier classifier
│   ├── embeddings.py     # 4 providers + auto-detect chain
│   ├── store.py          # ChromaDB wrapper, one collection per tier
│   ├── ingest.py         # batch ingest pipeline
│   ├── query.py          # hybrid vector + BM25 + RRF
│   ├── consolidate.py    # episodic → semantic distillation
│   ├── eval.py           # recall@K, MRR, latency benchmark
│   ├── memory.py         # ⭐ unified Memory facade (remember/recall/reflect/forget)
│   ├── watcher.py        # ⭐ real-time file-watcher daemon
│   ├── mcp_server.py     # ⭐ MCP stdio server
│   └── cli.py            # CLI entry point
├── tests/                # 55 unit + integration tests
├── benchmarks/           # golden.jsonl for eval
├── scripts/              # install.sh, cron.sh (legacy)
├── docs/                 # ARCHITECTURE.md, RESEARCH.md
├── data/                 # gitignored: chroma, logs, watcher state
└── .env.example          # config template
```

## Testing

```bash
./.venv/bin/pytest -q                   # all tests (419 as of v0.8.0)
./.venv/bin/pytest tests/test_memory.py # just the Memory facade
./.venv/bin/pytest -k "watcher"         # watcher tests
./.venv/bin/pytest -k "chroma"          # Chroma backend + enhancements
```

## Cross-platform support (macOS / Linux / Windows)

All Python code uses `pathlib` (no `os.path.join` string concat), so the
core brain works identically on all three OSes. Two scripts are
OS-specific:

| OS | Pre-commit hook script | Install command |
|---|---|---|
| macOS / Linux | `scripts/secret-scan.sh` | `ln -sf ../../scripts/secret-scan.sh .git/hooks/pre-commit` (already done in this repo) |
| Windows 10/11 | `scripts/secret-scan.ps1` | `pwsh scripts/install-pre-commit.ps1` |

Both scripts have the same patterns, the same exit codes, and the same
opt-out env var (`DUCKBOT_SKIP_SECRET_SCAN=1`).

### Windows quirks

- **Chromadb** ships pre-built wheels for Windows (Python 3.9-3.13). No
  compiler needed. `pip install chromadb` works out of the box.
- **`sentence-transformers`** (L7 rerank) also ships Windows wheels.
- **Long paths**: if your repo lives deep in the filesystem, Windows'
  260-char path limit can bite. Either move the repo closer to `C:\` or
  enable long paths in Group Policy.
- **Hugging Face Hub auth**: if you see a warning about
  `unauthenticated requests to the HF Hub`, set
  `HF_TOKEN=<your_token>` in `.env` to silence it. The model downloads
  work without a token (slower).
- **PowerShell version**: PowerShell 5.1 (ships with Windows 10) works
  for `secret-scan.ps1`. PowerShell 7+ is recommended and supports
  cross-platform pwsh invocation.

### Chroma `compact` (v0.8.0+)

Chroma's SQLite WAL mode grows unboundedly. The new `compact` CLI
reclaims that space:

```bash
./.venv/bin/python -m src.cli compact
# Compacting Chroma store at .../data/chroma...
#   [working] 90 chunks, no duplicates
#   [episodic] 3617 chunks, no duplicates
#   [semantic] 183 chunks, no duplicates
#   [procedural] 194 chunks, no duplicates
#   VACUUM'd .../chroma.sqlite3
# ✓ Compact complete: 0 duplicates removed, 4084 chunks kept
#   Disk: 70.7 MB → 60.3 MB (saved 10.4 MB)
```

### Chroma `distance_metric` (v0.8.0+)

HNSW distance metric is now configurable:

```bash
# Default: cosine (works for any embedding)
./.venv/bin/python -m src.cli query "..."

# Inner product — faster for pre-normalized vectors (BGE w/ normalize_embeddings=True)
DUCKBOT_CHROMA_DISTANCE=ip ./.venv/bin/python -m src.cli query "..."

# L2 — Euclidean, for raw (un-normalized) vectors
DUCKBOT_CHROMA_DISTANCE=l2 ./.venv/bin/python -m src.cli query "..."
```

**Note**: Chroma's `hnsw:space` only takes effect on collection
CREATION. Changing the metric on an existing store requires a new
persist dir or `compact`+reset. See the `DUCKBOT_CHROMA_DISTANCE` env
var in `src/store.py` for the dispatch.

## License

MIT — see LICENSE.
