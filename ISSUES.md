# Known Issues & Design Notes

## 1. Graceful Restart (FIXED in v0.15.2)

**Problem:** After a `brain_update` (git pull), the Python MCP server still runs the old code.
The only way to pick up changes was restarting the entire OpenClaw gateway.

**Root Cause:** The Python MCP server runs as a long-lived subprocess spawned by the
plugin shim. There was no mechanism to restart it without killing the gateway.

**Fix (v0.15.2):**
- `brain_restart` tool: sets `_restart_requested` flag → main loop exits cleanly
- Plugin shim `child.on('exit')`: auto-respawns a new Python process with a 2s delay
- Plugin shim `globalThis.handle.restart()`: manual restart without gateway restart
- Suppresses `NotOpenSSLWarning` urllib3 noise from stderr log

**Usage:**
```bash
# Via tool call (from any agent):
brain_restart({})

# Via plugin handle (from gateway logs or another process):
# globalThis[Symbol.for('openclaw.duckbot-memory')].restart()
```

---

## 2. Double MCP Servers (Known Behavior)

**Symptom:** `ps aux | grep mcp_server` shows 2 Python MCP server processes running.

**Root Cause:** OpenClaw gateway runs multiple worker threads (one per channel/instance).
Each worker independently loads the duckbot-memory plugin, which spawns its own Python
subprocess. This is expected — all instances share the same SQLite database.

**Status:** Not a bug. Both instances are fully functional and share state via the
same `data/brain.db` SQLite file. Closing one instance doesn't affect the other.

**Note:** If this causes issues (e.g., rate limits hit twice), the plugin should be
converted to a singleton pattern where only one Python process runs globally and all
gateway workers connect to it via IPC.

---

## 3. Workspace venv Uses System Python (Minor)

**Symptom:** `.venv/bin/python` is a symlink to `/Library/Developer/CommandLineTools/...python3.9`
instead of a proper venv interpreter.

**Root Cause:** The venv creation on macOS with CommandLineTools Python creates symlinks
rather than copying the interpreter.

**Impact:** Low — works fine in practice. Both workspace and desktop installs have this quirk.

**Fix if needed:** Recreate the venv using a standard Homebrew Python:
```bash
python3 -m venv --clear .venv
```

---

## 4. Desktop vs Workspace Repo Duality

**Symptom:** Two installations of duckbot-rag-memory exist:
- `~/Desktop/duckbot-rag-memory/` (legacy, now orphaned)
- `~/.openclaw/workspace/duckbot-rag-memory/` (current, tracked by OpenClaw plugin)

**Root Cause:** The repo was originally on Desktop. Later moved to the workspace directory
for proper OpenClaw integration.

**Fix applied:** LaunchAgent (`com.duckbot.memory-watcher`) updated to point to workspace repo.

**Recommendation:** Delete `~/Desktop/duckbot-rag-memory/` to avoid confusion, or keep as backup.

---

## 5. LaunchAgent Spawn Loop (Historical — Resolved)

**Symptom (historical):** Killing the watcher process caused it to respawn immediately.

**Root Cause:** The `com.duckbot.memory-watcher` LaunchAgent has `KeepAlive: {SuccessfulExit: false}`,
which means it always restarts the watcher, even for clean exits.

**Status:** This is intentional for the watcher daemon (it should always run). However, when
switching between repo versions (desktop ↔ workspace), manually kill all watchers first:
```bash
pkill -f 'src.watcher'
```
Then restart the LaunchAgent:
```bash
launchctl unload ~/Library/LaunchAgents/com.duckbot.memory-watcher.plist
launchctl load ~/Library/LaunchAgents/com.duckbot.memory-watcher.plist
```

---

## 6. brain_sync Hang on sentence_transformers (RESOLVED in v0.15.1)

**Symptom (resolved):** `brain_sync` tool call hung indefinitely on second+ call.

**Root Cause:** `sentence_transformers` CrossEncoder import hangs indefinitely when
`torch` is already loaded in the process. The import lock is uninterruptible.

**Fix (v0.15.1):**
- `_ensure_model()` refuses to attempt CrossEncoder load unless `sentence_transformers`
  is already in `sys.modules`
- ThreadPoolExecutor with 30s timeout for `be.score()` calls
- `SentenceTransformersBackend.available()` uses `find_spec` instead of `import`
- Cross-encoder rerank effectively disabled in environments with the torch hang
- Falls through to input order (acceptable — same ranking, no hang)

---

## 7. SIGSEGV in watcher.py (RESOLVED in v0.15.1)

**Symptom (resolved):** Watcher daemon crashed with SIGSEGV when using
`TierAssignment` as dict keys.

**Root Cause:** `TierAssignment` was a `@dataclass(frozen=True)` but without an explicit
`__hash__` override, making it unhashable. Using it as a dict key caused a segfault.

**Fix:** `TierAssignment` now has `@dataclass(eq=False, frozen=True)` + explicit `__hash__`.

---

## 8. Watcher RuntimeWarning (RESOLVED in v0.15.1)

**Symptom (resolved):** `RuntimeWarning: coroutine was never awaited` in watcher logs.

**Root Cause:** `store.add_chunks()` is an async function but was called without `await`.

**Fix:** Added `await` to the `store.add_chunks()` call in watcher.py.

---

## Future Improvements

1. **Singleton MCP server**: Only one Python process regardless of gateway worker count
2. **Plugin hot-reload**: Reload plugin code without restarting gateway
3. **LM Studio embeddings on Mac mini**: Currently using local sentence-transformers
   because LM Studio runs on Windows PC and is not reachable from Mac mini
4. **FSEvents watcher**: Currently polling (every 5 min); FSEvents available but
   disabled — set `DUCKBOT_WATCH_USE_FSEVENTS=1` to enable native macOS file watching
