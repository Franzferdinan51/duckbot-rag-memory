# Integration Guide

How to plug duckbot-rag-memory into the rest of your stack — Hermes,
OpenClaw, Claude Code, Codex, or anything else that speaks MCP stdio.

## TL;DR — for the impatient

```bash
# OpenClaw: use the native Node.js plugin (zero deps, auto session hooks)
./scripts/openclaw-bootstrap.sh
openclaw gateway restart
openclaw plugins list | grep duckbot-memory    # should show "✓ installed"

# Hermes: use the MemoryProvider plugin (auto-wakes on session start)
./scripts/hermes-bootstrap.sh
grep duckbot-brain ~/.hermes/config.yaml       # memory.provider: duckbot-brain

# Anything else (Claude Code / Cursor / Codex / mcporter): MCP server via launcher
hermes mcp add duckbot-memory \
  --command "$HOME/Desktop/duckbot-rag-memory/scripts/duckbot-memory-mcp.sh"
# Windows: --command "C:\Users\franz\Desktop\duckbot-rag-memory\scripts\duckbot-memory-mcp.bat"
```

The launcher reads your `.env` so the LMSTUDIO_API_KEY stays out of hermes config
and out of `hermes mcp list` / `/api/mcp/servers` redaction leaks.

## Why a launcher script?

Hermes's MCP config stores `command` + `args` + `env` in YAML. Three problems:

1. **Secrets leak.** Putting `LMSTUDIO_API_KEY=***` in `env:` puts it in
   `config.yaml`, and the `/api/mcp/servers` endpoint redaction is not
   perfect — anyone with file-system access to the YAML reads the key.
2. **`${VAR}` doesn't expand.** Verified in `hermes-agent/tools/mcp_tool.py`:
   `_build_safe_env()` merges user env literally without interpolation.
3. **OS venv paths differ.** Windows venvs use `.venv/Scripts/python.exe`;
   macOS/Linux use `.venv/bin/python`. Hardcoding either breaks the other.

The launcher script (`.sh` + `.bat`) solves all three: loads `.env` itself,
sets `PYTHONUNBUFFERED=1`, detects the right venv path, and exec's.

## Install paths

### macOS / Linux

The shell script is POSIX-bash. It works as-is from a git-bash install
on Windows too if you symlink it under your PATH:

```bash
# Single-line add
hermes mcp add duckbot-memory \
  --command "$HOME/Desktop/duckbot-rag-memory/scripts/duckbot-memory-mcp.sh"

# Verify
hermes mcp list          # should show ✓ enabled
hermes mcp test duckbot-memory
```

### Windows

Hermes on Windows uses Windows-native subprocess for stdio MCP servers,
which means `.sh` scripts trigger `WinError 193 (%1 is not a valid
Win32 application)`. Use the `.bat` wrapper:

```powershell
# PowerShell
hermes mcp add duckbot-memory `
  --command "C:\Users\franz\Desktop\duckbot-rag-memory\scripts\duckbot-memory-mcp.bat"

# Verify
hermes mcp list
hermes mcp test duckbot-memory
```

```bash
# git-bash / MSYS
hermes mcp add duckbot-memory \
  --command "/c/Users/franz/Desktop/duckbot-rag-memory/scripts/duckbot-memory-mcp.bat"
```

### Manual install (no launcher)

If you want to skip the wrapper script and point hermes straight at the
Python interpreter:

```bash
# macOS / Linux
hermes mcp add duckbot-memory \
  --command "$HOME/Desktop/duckbot-rag-memory/.venv/bin/python" \
  --args "-m src.mcp_server" \
  --env "PYTHONPATH=$HOME/Desktop/duckbot-rag-memory"

# Windows (PowerShell)
hermes mcp add duckbot-memory `
  --command "$HOME\Desktop\duckbot-rag-memory\.venv\Scripts\python.exe" `
  --args "-m src.mcp_server" `
  --env "PYTHONPATH=$HOME\Desktop\duckbot-rag-memory"
```

But for the secrets-not-in-config reason, prefer the launcher.

## Common gotchas

### `--env` flag order

Hermes's `mcp add --args` uses `nargs=REMAINDER` — every flag AFTER
`--args` gets swept into the args list. Always put `--env` first:

```bash
# WRONG — --env KEY=VAL gets caught in --args
hermes mcp add foo --args "-m server" --env "KEY=VAL"

# RIGHT
hermes mcp add foo --env "KEY=VAL" --args "-m server"
```

### API key not loading

Symptom: `stats` returns `lmstudio_reachable: false` or 401 errors.

Checklist:
1. `LMSTUDIO_API_KEY` is in `.env` (not just your shell).
2. `.env` is at the repo root, not `~/`.
3. You restarted hermes after editing `.env` (env is loaded at MCP server
   start, not on every call).

### `WinError 193` on Windows

You're passing a `.sh` path to `hermes mcp add --command`. Hermes on
Windows uses `CreateProcess` which only handles `.exe`/`.bat`/`.cmd`.
Use the `.bat` wrapper.

### Tools not appearing in session

`hermes mcp list` shows `✓ enabled` but you don't see the tools in the
active session. Hermes loads MCP tools at session start — start a new
session to pick them up.

## Verifying the integration

Three levels of confidence:

```bash
# 1. Process spawns cleanly
hermes mcp test duckbot-memory

# 2. Tools discovered
hermes mcp list  # should show 64 tools, ✓ enabled

# 3. End-to-end
python -m src.cli query "What did we decide about cloud-only models?"
```

If all three pass, you're live. The brain will surface relevant context
to your agent within ~0.6s per query.

## Enhanced Brain: Memory That Writes Back

The enhanced brain is what makes duckbot-rag-memory more than just storage.
It actively maintains the context files that agents read on startup.

### The problem it solves

Agents start each session with no memory of past work unless you manually
feed context. `brain_inflate` fetches relevant memories on-demand.
`brain_sync` writes memories back to the files agents read at startup.
Together they form a closed loop:

```
remember() → recall() → brain_inflate() → agent context
                        ↕
               brain_sync() ← reflect() + cron
```

### `brain_inflate` — On-demand context

When you start a new task or session, call `brain_inflate`:

```
# OpenClaw / Hermes tool call
brain_inflate(query="what are we working on?", k=10, agent_name="mavis")
```

Returns a markdown block ready to paste into your thinking — tier labels,
importance bars, source attribution:

```markdown
## 🧠 Mavis's Enhanced Memory Context

### 🧩 Semantic — Facts, concepts, knowledge
- [▓▓▓▓▓░░░░░] The API uses JWT for auth, not session cookies  _source: memory/soul.md_

### ⚙️ Procedural — Rules, patterns, how-to
- [▓▓▓▓▓▓▓░░░] Run `make test` before every PR  _source: memory/MEMORY.md_
```

### `brain_sync` — Keep context files fresh

Syncs memories to agent context files automatically:

| Platform | Path | Format | Notes |
|----------|------|--------|-------|
| OpenClaw | `~/.openclaw/workspace/memory/MEMORY.md` | Rich markdown | No char limit |
| OpenClaw | `~/.openclaw/workspace/memory/USER.md` | Rich markdown | User facts |
| OpenClaw | `~/.openclaw/workspace/memory/SOUL.md` | Rich markdown | Procedural tier |
| Hermes | `~/.hermes/memories/MEMORY.md` | `§`-delimited | 2,200 char limit |
| Hermes | `~/.hermes/memories/USER.md` | `§`-delimited | 1,375 char limit |
| Hermes | `~/.hermes/SOUL.md` | Markdown | Global personality |

The cron already calls `brain_sync` after every ingest run. To sync manually:

```bash
# Both platforms
python -m src.cli sync --target both

# OpenClaw only
python -m src.cli sync --target openclaw

# Hermes only
python -m src.cli sync --target hermes

# Dry run (preview)
python -m src.cli sync --dry-run
```

### OpenClaw skill

The `duckbot-brain` skill teaches your OpenClaw agents to use the enhanced
brain. Install it:

```bash
mkdir -p ~/.openclaw/workspace/skills/duckbot-brain/
ln -sf ~/Desktop/duckbot-rag-memory/skills/duckbot-brain/SKILL.md \
  ~/.openclaw/workspace/skills/duckbot-brain/SKILL.md
```

The skill file lives at `skills/duckbot-brain/SKILL.md` in this repo.
It auto-discovers from `~/.openclaw/workspace/skills/`.

## Configuration knobs

| Env var | Default | Effect |
|---------|---------|--------|
| `DUCKBOT_EMBED_CACHE_SIZE` | `4096` | LRU cache for embed results. `0` = disabled. |
| `DUCKBOT_EMBED_RPM` | `60` | Token-bucket rate (requests per minute). |
| `DUCKBOT_WATCH_USE_FSEVENTS` | unset | macOS FSEvents watcher (chromadb+watchdog can segfault). |
| `LMSTUDIO_URL` | `http://127.0.0.1:1234/v1` | LM Studio base URL. |
| `LMSTUDIO_MODEL` | `text-embedding-embeddinggemma-300m` | Embedding model to load in LM Studio. |
| `LMSTUDIO_RERANK_MODEL` | `Qwen/Qwen3-Reranker-0.6B` | Reranker model to load when `DUCKBOT_RERANK=1`. |
| `LMSTUDIO_API_KEY` | `lm-studio` | Bearer token. LM Studio's recent builds require it. |
| `OPENCLAW_WORKSPACE` | `~/.openclaw/workspace` | Watch this dir + auto-ingest new files. |
| `OPENCLAW_MEMORY` | `<workspace>/memory` | Specifically watch this subdir. |
| `HERMES_HOME` | `~/.hermes` | Hermes home dir. When set, `brain_sync` also writes to Hermes context files. |

All are read at startup. Restart the MCP server (and any consuming hermes
session) after changing them.
