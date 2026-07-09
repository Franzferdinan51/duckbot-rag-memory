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

## 2. Double MCP Servers ~~(Known Behavior)~~ → FIXED in v0.15.2

**Previous Symptom:** `ps aux | grep mcp_server` showed 2 Python MCP server processes.

**Root Cause:** Each gateway worker independently loaded the plugin and spawned its own
Python subprocess.

**Fix (v0.15.2):** Singleton pattern — module-level globals (`_sharedRpc`, `_sharedChild`,
`_refCount`) are shared across all gateway workers. First worker spawns the Python
process; subsequent workers join the singleton. Reference counting ensures only the last
worker to shut down kills the process.

**Result:** `ps aux | grep mcp_server` now shows exactly **1** Python process regardless
of gateway worker count.

---

## 3. Workspace venv Uses System Python (Low Priority)

**Symptom:** `.venv/bin/python` symlinks to CommandLineTools Python instead of a
standalone copy.

**Impact:** Low — works fine in practice. Only noticeable if the system Python
version changes.

**Fix if needed:**
```bash
python3 -m venv --clear .venv
.venv/bin/pip install -r requirements.txt
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

1. ~~Singleton MCP server~~ — ✅ Done in v0.15.2
2. **Plugin hot-reload**: Reload plugin code without restarting gateway
3. **LM Studio embeddings on Mac mini**: Currently using local sentence-transformers
   because LM Studio runs on Windows PC and is not reachable from Mac mini
4. **FSEvents watcher**: Currently polling (every 5 min); FSEvents available but
   disabled — set `DUCKBOT_WATCH_USE_FSEVENTS=1` to enable native macOS file watching
5. **Clean up orphaned desktop repo**: `~/Desktop/duckbot-rag-memory/` is deprecated;
   the canonical location is `~/.openclaw/workspace/duckbot-rag-memory/`
