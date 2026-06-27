# AGENTS.md - DuckBot RAG + Memory System

## What this is

A persistent RAG (Retrieval-Augmented Generation) + memory system built for DuckBot's OpenClaw + Hermes workflows. Inspired by (and pulling from) mem0, Letta, Cognee, Hermes Agent, and the CoALA paper.

## Quick start for agents

```bash
# Install deps
cd ~/Desktop/duckbot-rag-memory
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env to set LMSTUDIO_API_KEY (recommended) or OPENAI_API_KEY

# Ingest current memory (one-shot backfill)
python -m src.cli ingest ~/.openclaw/workspace/memory

# Query from any shell (loads .env, cross-platform venv detection)
./duck-memory query "What did we decide about cloud-only models?"
./scripts/duckbot-ask "What did we decide about cloud-only models?"
./scripts/duckbot-ask -f compact -n 3 "Duckets correction style"
./scripts/brain-recall "BATMAN worker offline"   # alias for duckbot-ask

# Update to the latest version
./scripts/update.sh                              # macOS/Linux Terminal
# Windows: double-click scripts\update.bat

# Agents use the CLI directly (returns JSON):
python -m src.cli update                        # pull + upgrade deps + doctor
python -m src.cli update --dry-run             # check if updates available

# Real-time ingest via the watcher daemon (recommended over cron)
./scripts/start-watcher.sh                       # polls every 5 min, content-hash dedup

# Query from Python directly
python -m src.cli query "What did we decide about cloud-only models?"
python -m src.cli stats
python -m src.cli doctor

# Expose as MCP tools to Hermes Agent (or any MCP client)
hermes mcp add duckbot-memory --command "$(pwd)/scripts/duckbot-memory-mcp.sh"
# Or on Windows: --command "$(pwd)\scripts\duckbot-memory-mcp.bat"
# Loads .env itself so LMSTUDIO_API_KEY never enters config.yaml.

# Run benchmark eval
python -m src.cli eval benchmarks/golden.jsonl

# Run tests
pytest -v
```

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the deep dive.

**TL;DR:**
- 4-tier memory model (working / episodic / semantic / procedural) from the CoALA paper
- Recursive markdown-aware chunking (512 tok, 15% overlap)
- LM Studio (local) or OpenAI embeddings
- ChromaDB embedded for vector storage
- Hybrid retrieval: vector + BM25-style keyword search + Reciprocal Rank Fusion
- Eval harness: recall@K, MRR, p50/p95 latency
- File-watcher daemon polls every 5 min (content-hash dedup — no spam)
- MCP stdio server with 67 tools (recall/remember/forget/reflect/stats/...)
- Shell wrappers: `scripts/duckbot-ask` + `scripts/brain-recall` for any cron/script

## File layout

```
duckbot-rag-memory/
|-- .github/            # Issue/PR templates + GitHub Actions CI
|   |-- workflows/ci.yml
|   |-- ISSUE_TEMPLATE/{bug_report,feature_request}.yml
|   `-- PULL_REQUEST_TEMPLATE.md
|-- src/                # core code
|   |-- chunk.py          # markdown chunker
|   |-- tier.py           # CoALA tier classifier
|   |-- embeddings.py     # pluggable embedding providers + shared http client + LRU cache + rate limit
|   |-- store.py          # ChromaDB wrapper (one collection per tier)
|   |-- ingest.py         # chunk -> tier -> embed -> upsert
|   |-- query.py          # hybrid vector + BM25 + RRF
|   |-- consolidate.py    # episodic -> semantic distillation
|   |-- eval.py           # benchmark runner
|   |-- memory.py         # the unified Memory facade (remember/recall/reflect/forget/stats)
|   |-- mcp_server.py     # MCP stdio JSON-RPC server (67 tools)
|   |-- watcher.py        # polling daemon (5-min interval, content-hash dedup)
|   |-- cli.py            # python -m src.cli
|   |-- decay.py fsrs.py rerank.py tier_priors.py  # brain layers (L8/L9/L7/L11)
|   |-- blocks.py entities.py graph.py              # brain layers (L3/L2/L1)
|   |-- injection_scan.py                          # brain layer (L15)
|   |-- verbatim_text storage is in chunk.py metadata + VerbatimResult in connectors/base.py (L13)
|   |-- backends/         # chroma / lancedb / qdrant (pluggable)
|   `-- connectors/       # openclaw / active_memory / dreaming / learn
|-- extensions/         # native OpenClaw plugin (Node.js shim → Python MCP server, zero deps)
|-- tests/              # pytest suite (748 tests collected; CI skips LM Studio tests)
|-- benchmarks/         # golden.jsonl for eval
|-- scripts/            # install / start / cron / launchers / helpers
|   |-- install.{ps1,sh,linux.sh,macos.sh}        # bootstrap (venv, deps, .env)
|   |-- start-watcher.{ps1,sh,windows.bat}        # file-watcher daemon
|   |-- duckbot-memory-mcp.{sh,bat}               # MCP stdio launcher (loads .env)
|   |-- duckbot-ask, brain-recall.sh               # shell brain-query helpers
|   |-- secret-scan.{ps1,sh}                      # pre-commit secret guard
|   `-- _format_{snippet,compact}.py              # format helpers for duckbot-ask
|-- skills/             # OpenClaw skill manifests + plugins
|-- docs/               # ARCHITECTURE.md, INTEGRATION.md, RESEARCH.md
|-- data/               # gitignored: chroma db + watcher state + logs
|-- .gitignore          # excludes env, data, venv, memory/
|-- .pre-commit-config.yaml  # local hook: scripts/secret-scan.sh
|-- README.md           # one-pager
|-- CHANGELOG.md        # version history (start here for context)
|-- CONTRIBUTING.md     # how to contribute + project values
|-- SECURITY.md         # vuln disclosure policy
|-- LICENSE             # MIT
`-- AGENTS.md           # this file
```

## Cron schedule

**scripts/cron.sh is DEPRECATED as of v0.15.0** — moved to
`scripts/archive/cron.sh.deprecated`. Use the watcher daemon instead
(`scripts/start-watcher.{ps1,sh,bat}`).

The watcher has been the recommended path since v0.10:
- Polls every 5 minutes (vs cron every 90 minutes)
- Dedups by content hash (no re-ingesting identical files)
- Daemonized with `start_new_session` (no orphan processes)
- Logs to `data/watcher.log`

If you still want the original cron-style nightly batch (consolidate +
eval + sync), run these manually:
```bash
python -m src.cli reflect   # episodic → semantic consolidation
python -m src.cli eval benchmarks/golden.jsonl
python -m src.cli sync      # write to OpenClaw/Hermes context files
```
The script handles:
1. Ingest from `~/.openclaw/workspace/memory` + project docs
2. Consolidate episodic -> semantic (heuristic)
3. Run eval against `benchmarks/golden.jsonl`
4. Snapshot stats
5. Commit logs + benchmarks to local repo

Logs go to `data/logs/cron-YYYYMMDD-HHMMSS.log`.

## Integration with OpenClaw + Hermes

This project integrates with both. See [docs/INTEGRATION.md](docs/INTEGRATION.md)
for the full step-by-step.

**OpenClaw (cron + ingest source):**
- The watcher daemon ingests from `~/.openclaw/workspace/memory/`,
  `AGENTS.md`, `SOUL.md`, `USER.md`, `IDENTITY.md`, `TOOLS.md`.
- The `/goal` skill can call `python -m src.cli query "..."` to surface
  prior context.

**Hermes Agent (MCP server):**
- `hermes mcp add duckbot-memory --command "$(pwd)/scripts/duckbot-memory-mcp.sh"`
  (or `.bat` on Windows). 67 tools become available:
  `brain_recall`, `brain_remember`, `brain_forget`, `brain_decay_status`,
  `brain_fsrs_review`, `brain_dreaming_read`, `brain_dreaming_cycle`,
  `brain_learn`, `brain_active_memory`, `recall_verbatim`,
  `search_verbatim`, `stats`, `watch`, `doctor`, etc.
- The launcher (`duckbot-memory-mcp.sh` / `.bat`) loads `.env` itself
  so `LMSTUDIO_API_KEY` never ends up in `hermes config.yaml`.

**Any MCP client (Claude Code, Cursor, Codex):**
- Same launchers work; just point the MCP config at the script.
- See [docs/INTEGRATION.md § Manual install](docs/INTEGRATION.md) for
  the JSON shape for `mcp.json`, `claude_desktop_config.json`, etc.

## Cross-platform line endings

`.gitattributes` is the source of truth. All `*.sh` / `*.ps1` / `*.py` /
`*.json` / `*.md` files normalize to LF on commit. Windows editors that
default to CRLF (Notepad, older VS Code configs) won't poison the repo.
Run `git diff --check` after editing to catch any stragglers.

## Design constraints

- **Idempotent ingest.** Re-running on the same file produces the same chunk IDs (content hash).
- **Per-tier storage.** Working tier can be aged out; procedural never is.
- **Open-source first.** Every external dep + every design pattern traces back to a public project (see `docs/RESEARCH.md`).
- **No agent runtime.** This is just memory, not a replacement for OpenClaw or Hermes.
- **Honest limitations.** Documented in CHANGELOG.md and inline TODO comments.
- **No deletions.** Additive changes only. Refactors that rename need a migration path.
- **No secrets.** `.env` is gitignored. `scripts/secret-scan.{ps1,sh}` enforces it pre-commit.

## Adding a new feature

1. Write code in `src/<module>.py` (or create a new module).
2. Add unit tests in `tests/test_<module>.py` (run `pytest -v`).
3. Update `docs/ARCHITECTURE.md` if design changed.
4. Update `CHANGELOG.md` with the version + summary under "Unreleased".
5. Verify no secrets (`bash scripts/secret-scan.sh`).
6. Commit + push to `origin/main` (or open a PR — see CONTRIBUTING.md).

## Testing

```bash
pytest -v                  # all tests (512 collected)
pytest tests/test_chunk.py # one module
pytest -k "tier"           # filter by name
pytest tests/test_duckbot_ask.py --timeout=60  # integration tests (need LM Studio)
```

Coverage gaps (future work):
- LLM-based fact extraction (currently regex-only)
- Cross-encoder rerank pass
- Concurrent ingest safety (multiple writers)
- Eval history trend detection

## Cost tracking

| Operation | Cost | Frequency |
|-----------|------|-----------|
| Embed a chunk (1536d, ~500 tok) | $0.00001 | per ingest |
| Embed 50k chunks | ~$0.50 | per cron run |
| 12 cron runs/day | ~$6/day | daily |
| Monthly (365 cron runs) | ~$180/month | monthly |

Local embedding mode cuts this to zero but is ~5x slower.

## License

MIT. See LICENSE.
