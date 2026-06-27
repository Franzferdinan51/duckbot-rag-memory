# OpenClaw Shim Dynamic Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure the OpenClaw Node shim exposes every tool reported by the live Python MCP server instead of only a stale hardcoded subset.

**Architecture:** Keep the current eager static registration so OpenClaw has an immediate surface during startup, then after the MCP `initialize` + `tools/list` handshake, register any discovered tool names that were not already registered. Update OpenClaw extension docs/comments to the live 67-tool count.

**Tech Stack:** Node.js built-in test runner, mocked child process JSON-RPC framing, Python MCP `tools/list` E2E smoke.

---

### Task 1: Add Shim Regression

**Files:**
- Modify: `extensions/duckbot-memory/test/shim.test.js`

- [x] **Step 1: Add an async handshake test**

Add this test after the existing “spawns python and registers hooks + tools” test:

```javascript
test('shim registers tools discovered after MCP tools/list handshake', async (t) => {
  const cp = require('node:child_process');
  const fakeChild = makeFakeChild();
  t.mock.method(cp, 'spawn', () => fakeChild);

  const entry = loadShim();
  const api = makeFakeApi({ repoPath: '/fake/repo', pythonPath: '/fake/python' });
  entry.register(api);

  const before = api.registeredTools.map((tool) => tool.opts.name);
  assert.ok(!before.includes('brain_graph_query'), 'test requires graph tool to be absent from static bootstrap list');

  fakeChild.stdout.emit('data', frame({ jsonrpc: '2.0', id: 1, result: { capabilities: {} } }));
  await new Promise((resolve) => setImmediate(resolve));
  fakeChild.stdout.emit('data', frame({
    jsonrpc: '2.0',
    id: 2,
    result: { tools: [{ name: 'brain_graph_query' }, { name: 'brain_recall' }] },
  }));

  await new Promise((resolve) => setImmediate(resolve));

  const after = api.registeredTools.map((tool) => tool.opts.name);
  assert.ok(after.includes('brain_graph_query'), 'discovered graph tool should be registered');
  assert.equal(after.filter((name) => name === 'brain_recall').length, 1, 'existing tools should not be duplicated');
  assert.deepEqual(globalThis[GLOBAL_KEY].handshakeTools(), ['brain_graph_query', 'brain_recall']);
});
```

- [x] **Step 2: Run the failing Node test**

Run:

```bash
node --test extensions/duckbot-memory/test/shim.test.js
```

Expected before implementation: FAIL because `brain_graph_query` is discovered but not registered.

### Task 2: Register Discovered Tools

**Files:**
- Modify: `extensions/duckbot-memory/index.js`

- [x] **Step 1: Add `registerToolName()` helper**

Replace direct registration in the static loop with a helper that deduplicates:

```javascript
const registeredTools = [];
const registeredToolSet = new Set();

function registerToolName(name) {
  if (registeredToolSet.has(name)) return false;
  try {
    api.registerTool(makeToolFactory(name), { name });
    registeredTools.push(name);
    registeredToolSet.add(name);
    return true;
  } catch (e) {
    logger.debug('[duckbot-memory] registerTool(%s) skipped: %s', name, e.message);
    return false;
  }
}
```

- [x] **Step 2: Use the helper for the static bootstrap list**

Inside the hardcoded list loop:

```javascript
registerToolName(name);
```

- [x] **Step 3: Register post-handshake discoveries**

After `toolNames = tools.map((t) => t.name);` add:

```javascript
let newlyRegistered = 0;
for (const name of toolNames) {
  if (registerToolName(name)) newlyRegistered += 1;
}
```

Update the ready log to include both counts:

```javascript
logger.info(
  '[duckbot-memory] ready: %d tools discovered, %d newly registered, %d total registered',
  toolNames.length, newlyRegistered, registeredTools.length,
);
```

### Task 3: Update OpenClaw Extension Docs

**Files:**
- Modify: `extensions/duckbot-memory/README.md`
- Modify: `extensions/duckbot-memory/package.json`
- Modify: `scripts/openclaw-bootstrap.sh`

- [x] **Step 1: Replace stale tool counts**

Change current `64`, `66`, and `66 tools` present-tense OpenClaw extension claims to `67`.

- [x] **Step 2: Clarify dynamic discovery**

In `extensions/duckbot-memory/README.md`, keep a sentence that says the shim eagerly registers a bootstrap surface and then registers any additional tools discovered from `tools/list`.

### Task 4: Verify End To End And Push

**Files:**
- Modify: `extensions/duckbot-memory/index.js`
- Modify: `extensions/duckbot-memory/test/shim.test.js`
- Modify: `extensions/duckbot-memory/README.md`
- Modify: `extensions/duckbot-memory/package.json`
- Modify: `scripts/openclaw-bootstrap.sh`
- Add: `docs/superpowers/plans/2026-06-27-openclaw-shim-dynamic-tools.md`

- [x] **Step 1: Run Node shim tests**

Run:

```bash
node --test extensions/duckbot-memory/test/shim.test.js
```

Expected: all shim tests pass.

- [x] **Step 2: Run real Python MCP tools/list E2E**

Run:

```bash
printf '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}\n' \
  | ./.venv/bin/python -m src.mcp_server >/tmp/duckbot-mcp-list.out
./.venv/bin/python - <<'PY'
import json
from pathlib import Path
data = json.loads(Path("/tmp/duckbot-mcp-list.out").read_text())
assert len(data["result"]["tools"]) == 67
print(len(data["result"]["tools"]))
PY
```

Expected: prints `67`.

- [x] **Step 3: Run full verification**

Run:

```bash
./.venv/bin/python -m pytest -q
./.venv/bin/python -m compileall -q src
git diff --check
bash scripts/secret-scan.sh
```

Expected: all pass.

- [ ] **Step 4: Commit and push**

Run:

```bash
git add extensions/duckbot-memory/index.js extensions/duckbot-memory/test/shim.test.js extensions/duckbot-memory/README.md extensions/duckbot-memory/package.json scripts/openclaw-bootstrap.sh docs/superpowers/plans/2026-06-27-openclaw-shim-dynamic-tools.md
git commit -m "Register OpenClaw shim tools from MCP discovery"
git push origin main
```
