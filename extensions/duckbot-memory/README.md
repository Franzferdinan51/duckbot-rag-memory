# DuckBot Memory — OpenClaw native plugin

A pure Node.js shim that wires [duckbot-rag-memory](../..) into OpenClaw as
a **natively-installed plugin**. Spawns the existing Python MCP server
(`src/mcp_server.py`, 68 tools) as a subprocess and proxies tool calls
over JSON-RPC over stdio. Zero npm dependencies; Node stdlib only.

> **Why a shim?** OpenClaw plugins run in-process inside the Node gateway
> (`openclaw/openclaw/src/plugins/loader.ts`). Python isn't supported
> natively. So we spawn the existing Python MCP server as a subprocess
> and bridge the 68 tools via JSON-RPC. No code duplication — the
> Python `src/mcp_server.py` IS the brain. The shim is pure glue.

## What you get

- **Native OpenClaw install.** Drop this directory into
  `~/.openclaw/extensions/duckbot-memory/` (or run the bootstrap which
  symlinks it). OpenClaw's plugin loader reads `openclaw.plugin.json`
  and loads `index.js` via `package.json#main`.
- **67 MCP tools** registered via `api.registerTool(factory, { name })`.
  The shim eagerly registers a bootstrap surface during startup, then
  registers any additional tools reported by `src.mcp_server.py`'s
  `tools/list` handshake.
- **`session_start` hook** fires `brain_wake_up` automatically on every
  session start and injects the result into the system prompt (so the
  agent starts with full context, no manual call needed).
- **`session_end` hook** fires `brain_sync --target openclaw` to write
  high-importance session facts back to OpenClaw's `MEMORY.md` /
  `USER.md` / `SOUL.md`.

## Install

### Option A — bootstrap script (recommended)

```bash
./scripts/openclaw-bootstrap.sh
```

This symlinks `extensions/duckbot-memory/` into `~/.openclaw/extensions/`,
auto-writes the config, and prints the activation command.

### Option B — manual

```bash
mkdir -p ~/.openclaw/extensions
ln -s "$(pwd)/extensions/duckbot-memory" ~/.openclaw/extensions/duckbot-memory

# Then add to ~/.openclaw/openclaw.json:
#   "plugins": { "entries": { "duckbot-memory": {
#       "enabled": true,
#       "config": { "repoPath": "/Users/you/Desktop/duckbot-rag-memory" }
#   } } }

openclaw gateway restart
openclaw plugins list | grep duckbot-memory     # should show "✓ installed"
```

## Plugin config

| Key | Default | Description |
|---|---|---|
| `repoPath` | (required) | Absolute path to your duckbot-rag-memory repo root. |
| `pythonPath` | `<repoPath>/.venv/bin/python` | Python interpreter inside the repo's venv. |
| `defaultK` | `5` | Top-K for `brain_recall` when the agent doesn't specify. |
| `autoWakeUp` | `true` | Fire `brain_wake_up` automatically on `session_start`. |
| `autoSync` | `true` | Fire `brain_sync` on `session_end`. |
| `timeoutMs` | `15000` | Per-tool-call timeout (ms). Stuck calls return an error to keep the agent loop from stalling. |

## How it works

```
OpenClaw gateway (Node.js)
  │
  │  registerHook('session_start', ...) → api calls our handler
  │  registerTool(name, factory)       → 68 tools registered
  │
  ▼
extensions/duckbot-memory/index.js   ← THIS SHIM (~250 lines, zero deps)
  │
  │  spawn(pythonPath, ['-u', '-m', 'src.mcp_server'], { cwd: repoPath })
  │
  ▼
src/mcp_server.py (Python, 68 tools)
  │
  ▼
ChromaDB + LM Studio + SQLite
```

The shim:
1. Spawns the Python MCP server as a subprocess with `child_process.spawn`.
2. Speaks JSON-RPC 2.0 over stdio using MCP's `Content-Length:` framing
   (falls back to newline-delimited JSON if the server uses that).
3. On `initialize` handshake, fetches the tool list and registers every
   tool via `api.registerTool(factory, { name })`. The factory builds an
   `AnyAgentTool` whose `execute()` writes a `tools/call` JSON-RPC
   message and awaits the response.
4. On `session_start` hook, fires `brain_wake_up` and pipes the result
   into `api.runtime.llm.injectSystemPrompt(...)` so the agent sees it.
5. On `session_end` hook, fires `brain_sync --target openclaw` to flush.
6. On `gateway_stop` hook, sends `shutdown` notification + SIGTERM to the
   Python subprocess for clean exit.

## Diagnostics

```bash
# Confirm the plugin loaded
openclaw plugins list | grep duckbot-memory

# Inspect the live shim state from inside a Node REPL
node -e "console.log(globalThis[Symbol.for('openclaw.duckbot-memory')])"

# Tail Python stderr (logged via api.logger → OpenClaw gateway log)
tail -f ~/.openclaw/gateway.log | grep duckbot-memory

# v0.15.1: Python stderr is also appended to data/mcp.log so segfaults
# and tracebacks survive after the gateway tears down. Set
# DUCKBOT_MCP_LOG=/custom/path.log to override, or DUCKBOT_MCP_LOG=""
# to disable entirely.
tail -f ~/Desktop/duckbot-rag-memory/data/mcp.log
```

## Testing

```bash
node --test extensions/duckbot-memory/test/
```

See `test/` for unit tests (mocked subprocess; no live Python needed).

## Compatibility

- OpenClaw ≥ 2026-06 (any build with the `definePluginEntry` SDK).
- Node.js ≥ 18 (uses `node:child_process`, `node:fs` stdlib only).
- Python 3.12+ (the project's minimum).
- LM Studio running on `127.0.0.1:1234` (default) — or set
  `DUCKBOT_EMBEDDING=openai` + `OPENAI_API_KEY` etc.

## Pattern sources

- [openclaw/openclaw `extensions/voice-call/index.ts`](https://github.com/openclaw/openclaw/blob/main/extensions/voice-call/index.ts) — canonical `definePluginEntry` shape.
- [openclaw/openclaw `docs/plugins/manifest.md`](https://github.com/openclaw/openclaw/blob/main/docs/plugins/manifest.md) — `openclaw.plugin.json` schema.
- [Model Context Protocol spec — stdio transport](https://spec.modelcontextprotocol.io/specification/basic/transports/) — `Content-Length:` framing.

## License

MIT — DuckBot brain contributors.
