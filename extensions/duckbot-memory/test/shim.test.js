/**
 * Unit tests for the duckbot-memory OpenClaw shim.
 *
 * Strategy:
 *   1. Mock `child_process.spawn` so we never actually launch Python.
 *   2. Build a fake `api` object that records `registerTool`/`registerHook`.
 *   3. Load the shim, call `register(api)`, assert the right hooks/tools
 *      were registered and that spawn was called with correct args.
 *   4. Separately exercise `StdioJsonRpc` with synthetic bytes to verify
 *      Content-Length framing, error handling, timeouts, exit cleanup.
 *
 * No live Python needed — tests run with `node --test`.
 */

'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { EventEmitter } = require('node:events');

const { StdioJsonRpc } = require('../lib/stdio-rpc');
const GLOBAL_KEY = Symbol.for('openclaw.duckbot-memory');

// ---- helpers --------------------------------------------------------------

/** Build a fake OpenClaw `api` that records every register call. */
function makeFakeApi(pluginConfig = {}) {
  const api = {
    logger: makeFakeLogger(),
    pluginConfig,
    runtime: { llm: { injectSystemPrompt: () => {} } },
    registeredTools: [],
    registeredHooks: [],
  };
  api.registerTool = (factory, opts) => {
    api.registeredTools.push({ opts, factory: typeof factory });
  };
  api.registerHook = (name, handler) => {
    api.registeredHooks.push({ name, handler });
  };
  return api;
}

function makeFakeLogger() {
  const log = { debug_msgs: [], info_msgs: [], warn_msgs: [], error_msgs: [] };
  log.debug = (...a) => log.debug_msgs.push(a.join(' '));
  log.info  = (...a) => log.info_msgs.push(a.join(' '));
  log.warn  = (...a) => log.warn_msgs.push(a.join(' '));
  log.error = (...a) => log.error_msgs.push(a.join(' '));
  return log;
}

/** Fake child process (EventEmitter with stdin/stdout/stderr stubs). */
function makeFakeChild() {
  const child = new EventEmitter();
  child.stdin = { write: () => true, end: () => {} };
  child.stdout = new EventEmitter();
  child.stderr = new EventEmitter();
  child.kill = (sig) => { child.killed = sig; child.emit('exit', null, sig); };
  child.pid = 99999;
  return child;
}

function loadShim() {
  delete require.cache[require.resolve('../index.js')];
  return require('../index.js');
}

/** Wrap a JSON-RPC message in Content-Length framed bytes. */
function frame(msg) {
  const body = Buffer.from(JSON.stringify(msg), 'utf8');
  return Buffer.concat([
    Buffer.from(`Content-Length: ${body.length}\r\n\r\n`, 'ascii'),
    body,
  ]);
}

// ---- tests ---------------------------------------------------------------

test('shim exports a definePluginEntry-shaped object', () => {
  const entry = loadShim();
  assert.equal(typeof entry, 'object');
  assert.equal(entry.id, 'duckbot-memory');
  assert.equal(typeof entry.register, 'function');
  assert.ok(entry.configSchema, 'configSchema required');
});

test('shim refuses to register without repoPath', () => {
  const entry = loadShim();
  const api = makeFakeApi({});
  entry.register(api);
  assert.equal(api.registeredTools.length, 0);
  assert.equal(api.registeredHooks.length, 0);
  assert.ok(
    api.logger.error_msgs.some((m) => /repoPath is required/.test(m)),
    'expected an error log for missing repoPath',
  );
});

test('shim spawns python and registers hooks + tools when configured', (t) => {
  const cp = require('node:child_process');
  const fakeChild = makeFakeChild();
  t.mock.method(cp, 'spawn', () => fakeChild);

  const entry = loadShim();
  const api = makeFakeApi({ repoPath: '/fake/repo', pythonPath: '/fake/python' });
  entry.register(api);

  // ---- spawn args -------------------------------------------------------
  assert.equal(cp.spawn.mock.calls.length, 1);
  const [python, args, opts] = cp.spawn.mock.calls[0].arguments;
  assert.equal(python, '/fake/python');
  assert.deepEqual(args, ['-u', '-m', 'src.mcp_server']);
  assert.equal(opts.cwd, '/fake/repo');
  assert.equal(opts.env.PYTHONUNBUFFERED, '1');

  // ---- spot-check tools (not exact count — upstream can add more) ------
  const toolNames = api.registeredTools.map((t) => t.opts.name);
  for (const must of [
    'brain_wake_up', 'brain_recall', 'brain_remember',
    'brain_skills_list', 'brain_sync', 'brain_palace',
    'remember', 'recall', 'reflect', 'stats', 'doctor', 'watch',
  ]) {
    assert.ok(toolNames.includes(must), `expected tool '${must}' registered`);
  }

  // ---- hooks registered -------------------------------------------------
  const hookNames = api.registeredHooks.map((h) => h.name);
  for (const must of ['session_start', 'session_end', 'gateway_stop']) {
    assert.ok(hookNames.includes(must), `expected hook '${must}' registered`);
  }

  // ---- globalThis handle exposed for diagnostics ------------------------
  const handle = globalThis[GLOBAL_KEY];
  assert.ok(handle, 'globalThis handle missing');
  assert.equal(handle.pid, 99999);
  assert.equal(handle.repoPath, '/fake/repo');
  assert.equal(handle.pythonPath, '/fake/python');
  assert.ok(Array.isArray(handle.registeredTools()));
  assert.ok(Array.isArray(handle.handshakeTools()));

  // ---- gateway_stop hook kills the child cleanly ------------------------
  const stopHook = api.registeredHooks.find((h) => h.name === 'gateway_stop').handler;
  stopHook();
  assert.equal(fakeChild.killed, 'SIGTERM');
});

// ---- StdioJsonRpc tests ----------------------------------------------------

test('StdioJsonRpc: Content-Length framed response resolves promise', async () => {
  const child = makeFakeChild();
  const logger = makeFakeLogger();
  const rpc = new StdioJsonRpc(child, logger);

  const promise = rpc.send('tools/list', {});
  child.stdout.emit('data', frame({ jsonrpc: '2.0', id: 1, result: { tools: [{ name: 'brain_recall' }] } }));

  const result = await promise;
  assert.deepEqual(result, { tools: [{ name: 'brain_recall' }] });
  rpc.close();
});

test('StdioJsonRpc: error response rejects the promise', async () => {
  const child = makeFakeChild();
  const rpc = new StdioJsonRpc(child, makeFakeLogger());

  const promise = rpc.send('tools/call', { name: 'brain_recall', arguments: {} });
  child.stdout.emit('data', frame({
    jsonrpc: '2.0', id: 1, error: { code: -32603, message: 'internal error' },
  }));

  await assert.rejects(promise, /internal error/);
  rpc.close();
});

test('StdioJsonRpc: newline-delimited JSON fallback', async () => {
  const child = makeFakeChild();
  const rpc = new StdioJsonRpc(child, makeFakeLogger());

  const promise = rpc.send('tools/list', {});
  child.stdout.emit('data', Buffer.from(
    JSON.stringify({ jsonrpc: '2.0', id: 1, result: { ok: true } }) + '\n',
    'utf8',
  ));

  assert.deepEqual(await promise, { ok: true });
  rpc.close();
});

test('StdioJsonRpc: child exit rejects all pending requests', async () => {
  const child = makeFakeChild();
  const rpc = new StdioJsonRpc(child, makeFakeLogger());

  const p1 = rpc.send('tools/list', {});
  const p2 = rpc.send('tools/call', { name: 'x', arguments: {} });
  child.emit('exit', 1, null);

  await assert.rejects(p1, /terminated/);
  await assert.rejects(p2, /terminated/);
  rpc.close();
});

test('StdioJsonRpc: stderr logs to warn', () => {
  const child = makeFakeChild();
  const logger = makeFakeLogger();
  const rpc = new StdioJsonRpc(child, logger);

  child.stderr.emit('data', Buffer.from('Warning: chroma not found\n', 'utf8'));
  assert.ok(
    logger.warn_msgs.some((m) => /python stderr/.test(m) && /chroma/.test(m)),
    'expected python stderr to be logged at warn level',
  );
  rpc.close();
});

test('StdioJsonRpc: malformed Content-Length header discards buffer (no infinite loop)', () => {
  const child = makeFakeChild();
  const logger = makeFakeLogger();
  const rpc = new StdioJsonRpc(child, logger);

  // Has the `\r\n\r\n` terminator but the header line is garbage (no
  // Content-Length: token). The drain loop must detect this and discard
  // the buffer to avoid spinning forever.
  const garbage = Buffer.from('totally-bogus-header\r\n\r\n', 'ascii');
  child.stdout.emit('data', garbage);
  assert.ok(
    logger.warn_msgs.some((m) => /malformed/i.test(m)),
    'expected malformed-header warn log',
  );
  rpc.close();
});

test('StdioJsonRpc: server-initiated notification forwarded to listener', () => {
  const child = makeFakeChild();
  const rpc = new StdioJsonRpc(child, makeFakeLogger());

  let received = null;
  rpc.onMessage((msg) => { received = msg; });
  // Server notification — no `id` field.
  child.stdout.emit('data', frame({
    jsonrpc: '2.0', method: 'notifications/progress', params: { pct: 42 },
  }));

  assert.deepEqual(received, {
    jsonrpc: '2.0', method: 'notifications/progress', params: { pct: 42 },
  });
  rpc.close();
});

test('StdioJsonRpc: close() rejects all in-flight requests', async () => {
  const child = makeFakeChild();
  const rpc = new StdioJsonRpc(child, makeFakeLogger());

  const p1 = rpc.send('tools/list', {});
  const p2 = rpc.send('tools/call', { name: 'x', arguments: {} });
  rpc.close();

  await assert.rejects(p1, /shim closed/);
  await assert.rejects(p2, /shim closed/);
});

test('StdioJsonRpc: notify() is fire-and-forget (no promise)', () => {
  const child = makeFakeChild();
  const rpc = new StdioJsonRpc(child, makeFakeLogger());
  // Should not throw, should not return a promise.
  assert.equal(rpc.notify('shutdown', null), undefined);
  rpc.close();
});