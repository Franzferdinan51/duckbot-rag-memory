# INSTALL.md — one-shot install for OpenClaw / Hermes / any MCP client

This document is the canonical install recipe for an agent (or human
operator) that wants to wire this brain into OpenClaw or Hermes Agent.
It exists so an OpenClaw / Hermes agent that has been pointed at this
file can copy-paste its way to a fully-fed brain without reading the
README.

## 0. What this is

A 4-tier CoALA-style RAG + long-term memory system that dramatically
expands the default memory of OpenClaw / Hermes Agent (which is
otherwise just a chat-history buffer). The brain runs as an MCP stdio
server; the agent invokes it as tools.

## 1. Prerequisites

- Python 3.12+ (`python3 --version`)
- pip
- ~2 GB disk for the venv + ChromaDB index
- For embeddings: LM Studio (recommended), OR MiniMax / OpenAI API key, OR `pip install sentence-transformers` for local

No GPU required. No system packages beyond what Python itself needs.

## 2. One-command install

```bash
# Clone + install
git clone https://github.com/Franzferdinan51/duckbot-rag-memory.git
cd duckbot-rag-memory
./scripts/install.sh         # or pwsh scripts/install.ps1 on Windows

# Configure
cp .env.example .env
# Edit .env: set DUCKBOT_EMBEDDING=lmstudio (default) + LMSTUDIO_URL,
# OR DUCKBOT_EMBEDDING=minimax + MINIMAX_API_KEY, etc.

# Verify
.venv/bin/python -m src.cli doctor
```

The `install.sh` script:
- creates a `.venv` (cross-platform path detection)
- installs all requirements
- sets the shell wrappers executable
- runs the secret scan

## 3. Bootstrap the brain from the existing OpenClaw / Hermes corpus

```bash
# For OpenClaw users — ingests every .md in ~/.openclaw/workspace
./scripts/openclaw-bootstrap.sh

# For Hermes Agent users — ingests every .md in ~/.hermes/memories
./scripts/hermes-bootstrap.sh
```

Both scripts:
1. Run `python -m src.cli doctor` to confirm setup
2. `python -m src.cli ingest <workspace>` — pulls every markdown
   file into the brain, auto-detecting tier
3. `python -m src.cli sync --target <platform>` — writes the
   consolidated MEMORY.md / USER.md / SOUL.md back into the agent's
   workspace
4. Print the MCP registration command for the next step

Re-running is idempotent (content-hash dedup, no duplicates).

## 4. Register the brain as an MCP server

### OpenClaw — native plugin (recommended)

The OpenClaw bootstrap script auto-installs a native Node.js plugin at
`~/.openclaw/extensions/duckbot-memory/`. It spawns the existing Python
MCP server as a subprocess and proxies all 64 tools + session_start /
session_end hooks via JSON-RPC. Zero npm dependencies.

```bash
# Already done by the bootstrap script. Just restart the gateway:
openclaw gateway restart
openclaw plugins list | grep duckbot-memory    # should show "✓ installed"
```

The plugin auto-fires `brain_wake_up` on every `session_start` so the
agent gets full context without being told to call it manually, and
auto-fires `brain_sync` on every `session_end` so high-importance
session facts get written back to OpenClaw's MEMORY.md / USER.md /
SOUL.md.

### OpenClaw — MCP server (fallback)

If the native plugin doesn't suit your setup, register the MCP server
directly:

```bash
hermes mcp add duckbot-memory \
  --command "$HOME/duckbot-rag-memory/scripts/duckbot-memory-mcp.sh"
```

Edit `~/.openclaw/openclaw.json` and add under `mcp.servers`:

```json
{
  "duckbot-brain": {
    "command": "/Users/you/duckbot-rag-memory/.venv/bin/python",
    "args": ["-m", "src.mcp_server"]
  }
}
```

Restart OpenClaw / `mcporter list` should now show 64 tools.

### Hermes Agent — MemoryProvider plugin (recommended)

The Hermes bootstrap script copies `src/plugins/memory/duckbot_brain/`
into `~/.hermes/plugins/memory/duckbot_brain/` and auto-writes
`memory.provider: duckbot-brain` into `~/.hermes/config.yaml`. The
plugin's `on_session_start` hook fires `brain_wake_up` automatically;
`on_session_end` persists durable user rules into the procedural tier.

```bash
./scripts/hermes-bootstrap.sh
grep duckbot-brain ~/.hermes/config.yaml    # memory.provider: duckbot-brain
```

### Hermes Agent — MCP server (fallback)

If the MemoryProvider plugin doesn't suit your setup, register the MCP
server directly:

```bash
hermes mcp add duckbot-memory \
  --command "$HOME/duckbot-rag-memory/scripts/duckbot-memory-mcp.sh"
```

Add the pre-flight hook to your `~/.hermesrc` (or SessionStart hook
config) so the brain loads on every session start:

```bash
# ~/.hermesrc or the equivalent config
$HOME/duckbot-rag-memory/scripts/hermes-preflight.sh
```

And the post-flight hook so the brain consolidates every session:

```bash
$HOME/duckbot-rag-memory/scripts/hermes-postflight.sh
```

## 5. First session — verify the brain is wired

In a fresh OpenClaw / Hermes session, call:

```
brain_wake_up
```

If the brain is healthy, you should get back a JSON (or markdown) object
with:
- `memories`: top-k recent chunks (with tiers + importance)
- `blocks`: any "memory blocks" (the user/SOUL-style always-on text)
- `graph_summary`: entity graph (people / projects / files)
- `fsrs_review_queue`: chunks that are due for review
- `stats`: chunk counts per tier

If `memories` is empty, the bootstrap didn't ingest — re-run the
relevant bootstrap script.

## 6. Cron / scheduled tasks (recommended)

To keep the brain healthy without thinking about it, schedule these
on a daily / weekly cron:

```cron
# Daily at 03:00 — prune decayed memories below R=0.05
0 3 * * * cd ~/duckbot-rag-memory && .venv/bin/python -m src.cli decay --apply --retention-floor 0.05

# Weekly Sunday 04:00 — re-tune the FSRS forgetting curve
0 4 * * 0 cd ~/duckbot-rag-memory && .venv/bin/python -m src.cli optimize-fsrs --apply

# Every 5 minutes — incremental ingest via the watcher (default interval)
*/5 * * * * cd ~/duckbot-rag-memory && .venv/bin/python -m src.watcher once >>data/watcher.log 2>&1
```

Or use the service installers that wrap the watcher as a long-lived
daemon:

```bash
./scripts/install-macos.sh   # launchd
./scripts/install-linux.sh   # systemd user service
pwsh scripts/install.ps1     # Windows Task Scheduler
```

## 7. Environment variables (full list)

| Variable | Default | Purpose |
|---|---|---|
| `DUCKBOT_EMBEDDING` | (auto-detect) | `lmstudio` / `minimax` / `openai` / `local` |
| `LMSTUDIO_URL` | `http://127.0.0.1:1234/v1` | LM Studio endpoint |
| `LMSTUDIO_API_KEY` | `lm-studio` | LM Studio bearer token |
| `LMSTUDIO_MODEL` | `text-embedding-embeddinggemma-300m` | Embedding model id |
| `MINIMAX_API_KEY` | (none) | MiniMax bearer token (if using MiniMax) |
| `OPENAI_API_KEY` | (none) | OpenAI bearer token (if using OpenAI) |
| `DUCKBOT_FSRS_W20` | `0.9` | FSRS-6 forgetting-curve exponent. Tune per deployment. |
| `DUCKBOT_RERANK` | `0` | Set to `1` to enable cross-encoder rerank (needs `sentence-transformers`). |
| `DUCKBOT_DECAY` | `0` | Set to `1` to enable Ebbinghaus decay weighting in recall. |
| `DUCKBOT_TIER_PRIORS` | `0` | Set to `1` to enable per-tier recall priors. |
| `DUCKBOT_FSRS` | `0` | Set to `1` to enable FSRS retention in recall. |
| `DUCKBOT_KEYWORD_BOOST` | `1` | MemPalace-style keyword boost (default on, zero cost). |
| `DUCKBOT_TEMPORAL_BOOST` | `1` | MemPalace-style temporal-proximity boost (default on, zero cost). |
| `DUCKBOT_SPELLCHECK` | `1` | Lightweight common-typo fixer on ingest. |
| `DUCKBOT_WATCH_USE_FSEVENTS` | `0` | Use native FSEvents instead of polling (macOS). |
| `DUCKBOT_SKIP_SECRET_SCAN` | `0` | Set to `1` to skip the pre-commit secret scan. |
| `HERMES_HOME` | `~/.hermes/memories` | Hermes Agent workspace path (used by `hermes-bootstrap.sh`). |
| `OPENCLAW_HOME` | `~/.openclaw/workspace` | OpenClaw workspace path (used by `openclaw-bootstrap.sh`). |

## 8. Common failure modes

| Symptom | Fix |
|---|---|
| `doctor` reports no embedding provider | Set `DUCKBOT_EMBEDDING` + the corresponding API key/url. |
| MCP server starts but tools fail with "Event loop is closed" | You're inside another event loop; restart the host (OpenClaw / Hermes). |
| Recall returns nothing | The brain was bootstrapped into a different `data/` directory. Check `pwd` of the MCP server. |
| Cross-encoder rerank times out | Reduce `--max-rank` to keep the rerank set small (< 20). |
| FSRS review queue is empty | Means no chunks have `fsrs_stability_days` or `fsrs_last_review_ts` metadata. Re-ingest or run `optimize-fsrs` to populate. |
| Bootstrap script says "no .md files" | Wrong workspace path; set `OPENCLAW_HOME` or `HERMES_HOME` env var. |

## 9. What to do next

- Read `README.md` for a tour of all 64 tools.
- Read `docs/ARCHITECTURE.md` for the 4-tier CoALA model + hybrid retrieval details.
- Read `docs/RESEARCH.md` for the upstream-project lineage (mem0, Letta, MemPalace, Graphiti, py-fsrs) and what was borrowed.
- Read `CHANGELOG.md` for the version-by-version history.
