# INSTALL.md — one-shot install for Hermes / any MCP client

This document is the canonical install recipe for an agent (or human
operator) that wants to wire this brain into Hermes Agent (the live
runtime) or another MCP client. It exists so an agent that has been
pointed at this file can copy-paste its way to a fully-fed brain
without reading the README. OpenClaw sections are kept for legacy
installs only — the reference deployment serves Hermes + Juno
exclusively (owner's 2026-11-23 ruling) and no longer syncs OpenClaw.

## 0. What this is

A 4-tier CoALA-style RAG + long-term memory system that dramatically
expands the default memory of Hermes Agent (which is otherwise just a
chat-history buffer). The brain runs as an MCP stdio server; the agent
invokes it as tools.

## 1. Prerequisites

- Python 3.12+ (`python3 --version`)
- pip
- ~2 GB disk for the venv + ChromaDB index
- For the default local path: LM Studio with an embedding model and `qwen3-reranker-0.6b` loaded in the local model list. The code default embedder is `text-embedding-embeddinggemma-300m`; the reference deployment pins `text-embedding-nomic-embed-text-v1.5` (768-dim) in `.env`.
- **Fact extraction is the agent's job at write time.** When your agent extracts facts from a memory, pass them via `brain_remember(facts=[...])` or `reflect(extract_callback=...)`; `reflect()` uses lightweight regex heuristics when no agent facts are supplied. Separately, the deployed nightly consolidation job runs a dedicated writer model (`lfm3-omni-2b`, load-on-dispatch) over the day's episodic chunks — see the README's deployed-jobs table.
- For alternate embeddings: LM Studio (recommended), OR an OpenAI API key, OR `pip install sentence-transformers` for local. (MiniMax still exists as an explicit-only provider but was removed from auto-detect in v0.17.0 and is deprecated.)

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
# LMSTUDIO_MODEL=text-embedding-nomic-embed-text-v1.5 (the reference
# deployment's pin; the code default is embeddinggemma-300m),
# LMSTUDIO_RERANK_MODEL=qwen3-reranker-0.6b.
# OR DUCKBOT_EMBEDDING=openai + OPENAI_API_KEY, etc.

# Verify
.venv/bin/python -m src.cli doctor
```

The `install.sh` script:
- creates a `.venv` (cross-platform path detection)
- installs all requirements
- sets the shell wrappers executable
- runs the secret scan

## 3. Bootstrap the brain from the existing Hermes corpus

```bash
# For Hermes Agent users — ingests every .md in ~/.hermes/memories
./scripts/hermes-bootstrap.sh

# Legacy, for OpenClaw installs — ingests every .md in ~/.openclaw/workspace
./scripts/openclaw-bootstrap.sh
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

### OpenClaw — native plugin (legacy)

The OpenClaw bootstrap script auto-installs a native Node.js plugin at
`~/.openclaw/extensions/duckbot-memory/`. It spawns the existing Python
MCP server as a subprocess and proxies all 73 tools + session_start /
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

### OpenClaw — MCP server (fallback, legacy)

If the native plugin doesn't suit your setup, register the MCP server
directly in `~/.openclaw/openclaw.json` under `mcp.servers`:

```json
{
  "duckbot-brain": {
    "command": "/Users/you/duckbot-rag-memory/.venv/bin/python",
    "args": ["-m", "src.mcp_server"]
  }
}
```

Restart OpenClaw / `mcporter list` should now show 73 tools.

### Hermes Agent — MemoryProvider plugin (recommended)

The Hermes bootstrap script installs the plugin to the canonical
**top-level** path `~/.hermes/plugins/duckbot-brain/` and auto-writes
`memory.provider: duckbot-brain` into `~/.hermes/config.yaml`.
(That exact path is load-bearing — Hermes' memory-provider resolver
only looks at top-level `plugins/<provider-name>/`.) The plugin's
`on_session_start` hook fires `brain_wake_up` automatically;
`on_session_end` persists durable user rules into the procedural tier.

```bash
./scripts/hermes-bootstrap.sh
grep duckbot-brain ~/.hermes/config.yaml    # memory.provider: duckbot-brain
```

**Host-gate caveat (the reference Mac):** Hermes' gateway/backend
processes run an isolated (`-I`) bundled Python — no user
site-packages, `PYTHONPATH` stripped — so the plugin cannot import
`chromadb` and **self-disables** there. That is by design, not a bug:
the canonical Hermes surface on such hosts is the MCP server below, and
`hermes memory off` (unselects the provider) keeps the session startup
quiet. Where the Hermes Python *can* import chromadb, the in-process
provider works exactly as described above.

### Hermes Agent — MCP server (fallback)

If the MemoryProvider plugin doesn't suit your setup, register the MCP
server directly:

```bash
hermes mcp add duckbot-memory \
  --command "$HOME/duckbot-rag-memory/scripts/duckbot-memory-mcp.sh"
```

### Session hooks — already wired by the plugin

You do NOT need to wire `hermes-preflight.sh` / `hermes-postflight.sh`
into a Hermes config file. The MemoryProvider plugin declares its own
`on_session_start` + `on_session_end` hooks in `plugin.yaml` — those
fire automatically once the plugin is activated (via
`memory.provider: duckbot-brain` in `~/.hermes/config.yaml`, which the
bootstrap script writes for you).

| Hook | Behavior |
|---|---|
| `on_session_start` | Calls `brain.wake_up(k=8, ...)` and returns the wake-up shape to Hermes so the agent starts with full context — no manual call needed. |
| `on_session_end` | Regex-scans the session messages for durable user rules (always / never / must / should / prefer / want / don't, 12–800 chars) and persists them as a single `procedural`-tier chunk via `brain_remember`. |

Where the plugin self-disables (see the host-gate caveat above), these
hooks do not fire — the MCP `brain_wake_up` / `brain_remember` tools and
the nightly jobs in the README deployed-jobs table cover the same ground.

The `hermes-preflight.sh` + `hermes-postflight.sh` scripts still exist
for cron / manual use cases (e.g. nightly consolidation) — they're
NOT invoked automatically by Hermes. Use them like:

```bash
# Nightly cron: reflect over the last 7 days
$HOME/duckbot-rag-memory/scripts/hermes-postflight.sh --days 7

# One-shot wake-up anchored on a topic (paste output into agent context)
$HOME/duckbot-rag-memory/scripts/hermes-preflight.sh --query Hermes
```

## 5. First session — verify the brain is wired

In a fresh Hermes (or MCP client) session, call:

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

> **Reference deployment:** the live brain does not use cron — it runs
> systemd user timers on Robin (hourly sync, 03:00 decay, 03:30 nightly
> consolidation + graph backfill, Sun 04:00 FSRS, Sun 05:00 backup).
> See the README deployed-jobs table. The cron lines below are for a
> fresh single-host install.

To keep the brain healthy without thinking about it, schedule these
on a daily / weekly cron:

```cron
# Daily at 03:00 — prune decayed memories below R=0.05
0 3 * * * cd ~/duckbot-rag-memory && .venv/bin/python -m src.cli decay --apply --retention-floor 0.05

# Weekly Sunday 04:00 — re-tune the FSRS forgetting curve
0 4 * * 0 cd ~/duckbot-rag-memory && .venv/bin/python -m src.cli optimize-fsrs --apply

# Weekly Sunday 05:00 — full brain backup (keeps newest 4)
0 5 * * 0 cd ~/duckbot-rag-memory && bash scripts/brain-backup-wrapper.sh

# Every 5 minutes — incremental ingest via the watcher (default interval;
# optional — the reference deployment does not run the watcher)
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
| `DUCKBOT_EMBEDDING` | (auto-detect) | `lmstudio` / `openai` / `local` (`minimax` explicit-only, deprecated, never auto-detected) |
| `LMSTUDIO_URL` | `http://127.0.0.1:1234/v1` | LM Studio endpoint |
| `LMSTUDIO_API_KEY` | `lm-studio` | LM Studio bearer token |
| `LMSTUDIO_MODEL` | `text-embedding-embeddinggemma-300m` | Embedding model id (reference deployment pins `text-embedding-nomic-embed-text-v1.5`) |
| `LMSTUDIO_RERANK_MODEL` | `qwen3-reranker-0.6b` | Reranker model id used when `DUCKBOT_RERANK=1` |
| `DUCKBOT_GRAPH_FEED` | `0` | Set to `1` to feed every new memory into the entity graph (live on the reference deployment since v0.18.0) |
| `MINIMAX_API_KEY` | (none) | MiniMax bearer token (only if explicitly selecting the deprecated MiniMax provider) |
| `OPENAI_API_KEY` | (none) | OpenAI bearer token (if using OpenAI) |
| `DUCKBOT_FSRS_W20` | `0.9` | FSRS-6 forgetting-curve exponent. Tune per deployment. |
| `DUCKBOT_RERANK` | `0` | Set to `1` to enable cross-encoder rerank (LM Studio Qwen3 reranker or `sentence-transformers`). |
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
| Cross-encoder rerank times out | Keep `k` small for recall or disable rerank for that query; the reranker scores the current top results only. |
| FSRS review queue is empty | Means no chunks have `fsrs_stability_days` or `fsrs_last_review_ts` metadata. Re-ingest or run `optimize-fsrs` to populate. |
| Bootstrap script says "no .md files" | Wrong workspace path; set `OPENCLAW_HOME` or `HERMES_HOME` env var. |

## 9. What to do next

- Read `README.md` for a tour of all 73 tools.
- Read `docs/ARCHITECTURE.md` for the 4-tier CoALA model + hybrid retrieval details.
- Read `docs/RESEARCH.md` for the upstream-project lineage (mem0, Letta, MemPalace, Graphiti, py-fsrs) and what was borrowed.
- Read `CHANGELOG.md` for the version-by-version history.

## Remote brain host (Hermes on a different machine)

When Hermes runs on one machine and the brain store lives on another,
point Hermes' MCP entry at **scripts/duckbot-memory-mcp-client.sh** on the
Hermes host instead of a hand-written ssh command. The launcher ssh's to
the brain host with keepalives (ServerAliveInterval=20, CountMax=3) and a
connect timeout: if the far end vanishes (VPN/Tailscale flips, sleep,
network drop), ssh notices within ~60s and exits, so Hermes respawns a
fresh transport on its next brain call instead of hanging into a dead
pipe for days (the 2026-11-23 stall). Host, remote script path and key
are overridable via DUCKBOT_BRAIN_HOST / DUCKBOT_BRAIN_SCRIPT /
DUCKBOT_BRAIN_KEY.

## Importing another agent's memories (manual feature)

The brain's live connections are Hermes/Juno-only, but the owner can
import another agent's sessions on demand. Imports are NEVER automatic:
nothing is scheduled, and the tool refuses to run until a source is
named explicitly.

Run on the machine that holds the sessions and the writer model
(LM Studio), from the repo root:

    # grok-local or zcode-local session stores (last 30 days)
    .venv/bin/python scripts/ingest_sessions.py --sources grok --days 30
    .venv/bin/python scripts/ingest_sessions.py --sources zcode --days 30

    # any agent's exported files (.jsonl / .md / .txt, files or a folder)
    .venv/bin/python scripts/ingest_sessions.py --files ~/exports/some-agent/

Add --dry-run first to list what would be imported without storing
anything. Imports are state-tracked (per-file mtime) and content-hash
deduped, so re-running an import is a no-op for unchanged sessions.
Imported material goes through the same distillation pipeline as native
sessions (facts, episodes, intentions) with the session file recorded
as its source.
