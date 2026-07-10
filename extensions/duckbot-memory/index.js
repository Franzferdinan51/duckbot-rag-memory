/**
 * DuckBot Memory — OpenClaw native plugin (Node.js shim).
 *
 * Pure Node.js, zero npm dependencies. Spawns the Python MCP server
 * (duckbot-rag-memory/src/mcp_server.py) as a subprocess and proxies
 * 68 brain tools + session_start / session_end hooks into OpenClaw.
 *
 * Singleton pattern: a single Python subprocess is shared across all
 * gateway workers. The first worker to load this plugin spawns the Python
 * process; subsequent workers reuse it. Reference counting ensures the
 * process is only killed when the last worker shuts down.
 *
 * Pattern sources:
 *   - openclaw/openclaw `extensions/voice-call/index.ts` (definePluginEntry)
 *   - openclaw/openclaw `docs/plugins/manifest.md` (openclaw.plugin.json)
 *   - MCP spec (Content-Length framed JSON-RPC over stdio)
 *     https://spec.modelcontextprotocol.io/specification/basic/transports/
 */

'use strict';

const { spawn } = require('node:child_process');
const { existsSync } = require('node:fs');
const path = require('node:path');
const process = require('node:process');

const GLOBAL_KEY = Symbol.for('openclaw.duckbot-memory');
const { StdioJsonRpc } = require('./lib/stdio-rpc');

// ---------------------------------------------------------------------------
// Module-level singleton — shared across ALL gateway workers.
// First worker to load this plugin spawns the Python process.
// All subsequent workers reuse it. Reference-counted shutdown.
// ---------------------------------------------------------------------------
let _sharedRpc = null;       // StdioJsonRpc instance
let _sharedChild = null;      // child_process handle
let _sharedChildPid = null;
let _refCount = 0;           // number of workers using this process
let _initialized = false;     // MCP handshake complete
let _toolNames = [];         // tools from server/tools/list
let _registeredTools = [];    // names registered with the API
let _registeredToolSet = new Set();
let _stderrLogStream = null;
let _stderrLogPath = null;
let _initiatingShutdown = false;
let _spawnLogger = null;      // set on first spawn, used by later workers
let _repoPath = null;
let _pythonPath = null;
let _defaultK = 5;
let _autoWakeUp = true;
let _autoSync = true;
let _timeoutMs = 15000;
let _api = null;              // set on first load; used by reinitialize

// Deferred initialization queue — tools registered while spawning get queued
// and flushed once the handshake completes.
const _deferredTools = [];

// ---- spawn the shared Python subprocess ------------------------------------
function spawnSharedChild(logger) {
  _spawnLogger = logger;
  logger.info('[duckbot-memory] spawning shared Python MCP server (repo=%s, python=%s)', _repoPath, _pythonPath);

  const child = spawn(
    _pythonPath,
    ['-u', '-m', 'src.mcp_server'],
    {
      cwd: _repoPath,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: '1',
        DUCKBOT_EMBEDDING: process.env.DUCKBOT_EMBEDDING || 'lmstudio',
        DUCKBOT_REPO_PATH: _repoPath,
      },
      stdio: ['pipe', 'pipe', 'pipe'],
    },
  );
  _sharedChildPid = child.pid;

  child.on('error', (err) => {
    logger.error('[duckbot-memory] python child error (pid=%d): %s', _sharedChildPid, err.message);
  });

  // Tee stderr to log file, suppress urllib3 NotOpenSSLWarning noise.
  if (_stderrLogStream) {
    child.stderr.on('data', (chunk) => {
      const text = chunk.toString('utf8');
      if (text.includes('NotOpenSSLWarning')) return;  // cosmetic, noisy
      _stderrLogStream.write(chunk);
    });
  }

  // Exit handler — auto-respawn unless we're shutting down globally.
  child.on('exit', (code, signal) => {
    const reason = code != null ? `exit ${code}` : `signal ${signal}`;
    logger.info('[duckbot-memory] shared Python MCP server exited (%s, pid=%d, refs=%d)', reason, _sharedChildPid, _refCount);
    if (_initiatingShutdown) {
      logger.info('[duckbot-memory] global shutdown in progress — not respawning');
      return;
    }
    logger.info('[duckbot-memory] auto-respawning in 2s... (refs=%d)', _refCount);
    setTimeout(() => {
      // Respawn and reinitialize — the new RPC will be assigned to _sharedRpc.
      _sharedRpc = null;
      _initialized = false;
      spawnSharedChild(logger);
      reinitializeShared();
    }, 2000);
  });

  _sharedChild = child;
  _sharedRpc = new StdioJsonRpc(child, logger);

  return child;
}

// ---- reinitialize after a respawn -----------------------------------------
async function reinitializeShared() {
  const logger = _spawnLogger;
  const rpc = _sharedRpc;
  if (!rpc) return;
  try {
    await rpc.send('initialize', {
      protocolVersion: '2024-11-05',
      capabilities: {},
      clientInfo: { name: 'duckbot-memory-openclaw-shim', version: '0.1.0' },
    });
    rpc.notify('notifications/initialized', {});
    const { tools } = await rpc.send('tools/list', {});
    _toolNames = tools.map((t) => t.name);

    // Register any new tools the new server exposes.
    let newlyRegistered = 0;
    for (const name of _toolNames) {
      if (!_registeredToolSet.has(name)) {
        _registeredTools.push(name);
        _registeredToolSet.add(name);
        newlyRegistered++;
      }
    }

    _initialized = true;

    // Flush deferred tool registrations.
    while (_deferredTools.length > 0) {
      const { name, api: deferredApi } = _deferredTools.shift();
      try {
        deferredApi.registerTool(makeToolFactory(name, rpc), { name });
      } catch (e) { /* already registered or rejected */ }
    }

    logger.info(
      '[duckbot-memory] respawn complete: %d tools (pid=%d), deferred flushed',
      _toolNames.length, _sharedChildPid,
    );
  } catch (e) {
    logger.error('[duckbot-memory] reinitialize failed: %s', e.message);
  }
}

// ---- tool factory — closes over the RPC instance ---------------------------
function makeToolFactory(toolName, rpcInstance) {
  const activeRpc = rpcInstance || _sharedRpc;
  return () => ({
    name: toolName,
    description: `DuckBot brain tool: ${toolName}`,
    parameters: { type: 'object', properties: {}, additionalProperties: true },
    async execute(args) {
      if (!_initialized) {
        throw new Error('duckbot-memory shim still initializing; retry in a moment');
      }
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), _timeoutMs);
      try {
        const result = await activeRpc.send('tools/call', {
          name: toolName,
          arguments: args || {},
        });
        return result;
      } catch (e) {
        if (e.name === 'AbortError') {
          throw new Error(`duckbot-memory tool '${toolName}' timed out after ${_timeoutMs}ms`);
        }
        throw e;
      } finally {
        clearTimeout(timer);
      }
    },
  });
}

// ---- register a tool with the given API instance --------------------------
function registerTool(name, apiInstance) {
  if (_registeredToolSet.has(name)) return;
  if (!_initialized) {
    // Defer — queue for flush after handshake.
    _deferredTools.push({ name, api: apiInstance });
    return;
  }
  try {
    apiInstance.registerTool(makeToolFactory(name, _sharedRpc), { name });
    _registeredTools.push(name);
    _registeredToolSet.add(name);
  } catch (e) {
    _spawnLogger?.debug('[duckbot-memory] registerTool(%s) skipped: %s', name, e.message);
  }
}

// ---- fire brain_wake_up on session_start ---------------------------------
async function fireWakeUp(event) {
  if (!_autoWakeUp || !_initialized) return;
  try {
    const wake = await _sharedRpc.send('tools/call', {
      name: 'brain_wake_up',
      arguments: { k: _defaultK, include_blocks: true, include_graph: true, include_fsrs_review: true },
    });
    const text = wake?.content?.[0]?.text || '';
    if (text && _api?.runtime?.llm?.injectSystemPrompt) {
      _api.runtime.llm.injectSystemPrompt(
        `\n[duckbot-memory: session_start brain_wake_up]\n${text}\n`,
      );
    }
    _spawnLogger?.info('[duckbot-memory] session_start: brain_wake_up fired (session=%s)', event?.sessionId);
  } catch (e) {
    _spawnLogger?.warn('[duckbot-memory] session_start hook failed: %s', e.message);
  }
}

// ---- fire brain_sync on session_end --------------------------------------
async function fireSync(event) {
  if (!_autoSync || !_initialized) return;
  try {
    await _sharedRpc.send('tools/call', {
      name: 'brain_sync',
      arguments: { target: 'openclaw', dry_run: false },
    });
    _spawnLogger?.info('[duckbot-memory] session_end: brain_sync fired (session=%s, msgs=%d)',
        event?.sessionId, event?.messageCount);
  } catch (e) {
    _spawnLogger?.warn('[duckbot-memory] session_end hook failed: %s', e.message);
  }
}

// ---------------------------------------------------------------------------
// Plugin entry
// ---------------------------------------------------------------------------
function defaultPythonPath(repoPath) {
  if (process.platform === 'win32') {
    const p = path.join(repoPath, '.venv', 'Scripts', 'python.exe');
    return existsSync(p) ? p : 'python';
  }
  const p = path.join(repoPath, '.venv', 'bin', 'python');
  return existsSync(p) ? p : 'python3';
}

function definePluginEntry(opts) { return opts; }

module.exports = definePluginEntry({
  id: 'duckbot-memory',
  configSchema: require('./openclaw.plugin.json').configSchema,

  register(api) {
    const logger = api.logger;

    // ---- resolve config (first worker to load sets these globals) ----------
    if (_repoPath === null) {
      // First load — set up globals and spawn the shared process.
      _repoPath = api.pluginConfig?.repoPath || process.env.DUCKBOT_REPO_PATH;
      if (!_repoPath) {
        logger.error('[duckbot-memory] repoPath is required (set in plugin config or DUCKBOT_REPO_PATH env)');
        return;
      }
      _pythonPath = api.pluginConfig?.pythonPath || process.env.DUCKBOT_PYTHON_PATH || defaultPythonPath(_repoPath);
      _defaultK = Number.isFinite(api.pluginConfig?.defaultK) ? api.pluginConfig.defaultK : 5;
      _autoWakeUp = api.pluginConfig?.autoWakeUp !== false;
      _autoSync = api.pluginConfig?.autoSync !== false;
      _timeoutMs = Number.isFinite(api.pluginConfig?.timeoutMs) ? api.pluginConfig.timeoutMs : 15_000;

      // ---- stderr log ----------------------------------------------------
      const logPathEnv = process.env.DUCKBOT_MCP_LOG;
      if (logPathEnv !== '') {
        _stderrLogPath = logPathEnv || path.join(_repoPath, 'data', 'mcp.log');
        try {
          const fs = require('node:fs');
          fs.mkdirSync(path.dirname(_stderrLogPath), { recursive: true });
          _stderrLogStream = fs.createWriteStream(_stderrLogPath, { flags: 'a' });
          _stderrLogStream.on('error', (e) => {
            logger.warn('[duckbot-memory] stderr log stream error: %s', e.message);
          });
        } catch (e) {
          logger.warn('[duckbot-memory] could not open %s: %s', _stderrLogPath, e.message);
          _stderrLogStream = null;
        }
      }

      // Spawn — this is the one and only spawn for this entire process.
      spawnSharedChild(logger);
      _api = api;

      // Kick off the MCP handshake asynchronously.
      (async () => {
        try {
          await _sharedRpc.send('initialize', {
            protocolVersion: '2024-11-05',
            capabilities: {},
            clientInfo: { name: 'duckbot-memory-openclaw-shim', version: '0.1.0' },
          });
          _sharedRpc.notify('notifications/initialized', {});
          const { tools } = await _sharedRpc.send('tools/list', {});
          _toolNames = tools.map((t) => t.name);

          for (const name of _toolNames) {
            registerTool(name, api);
          }
          _initialized = true;

          // Flush deferred tools (unlikely to have any, but safe).
          while (_deferredTools.length > 0) {
            const { name, api: deferredApi } = _deferredTools.shift();
            registerTool(name, deferredApi);
          }

          logger.info(
            '[duckbot-memory] ready: %d tools, %d registered, pid=%d (singleton, refs=1)',
            _toolNames.length, _registeredTools.length, _sharedChildPid,
          );
        } catch (e) {
          logger.error('[duckbot-memory] MCP initialize failed: %s', e.message);
        }
      })();

    } else {
      // Subsequent load — just register tools against the existing process.
      logger.info('[duckbot-memory] joining singleton (existing pid=%d, refs=%d→%d)',
          _sharedChildPid, _refCount, _refCount + 1);
      _api = api;
    }

    _refCount++;

    // ---- register tools (eager — visible before handshake completes) -----
    const staticTools = [
      'brain_wake_up','brain_recall','brain_recall_verbatim','brain_remember',
      'brain_reflect','brain_stats','brain_fsrs_review','brain_decay_status',
      'brain_search_verbatim','brain_skills_list','brain_skills_suggest',
      'brain_skills_promote','brain_inflate','brain_sync','brain_index',
      'brain_nudge','brain_skill_create','brain_user_model','brain_palace',
      'brain_optimize_fsrs','brain_apply_fsrs_w20','brain_fsrs_optimize_apply',
      'brain_export','brain_import','brain_seed_demo','brain_restart',
      'remember','recall','reflect','forget','stats','watch','doctor',
      'recall_verbatim','fsrs_review','decay_status','forget_by_query',
      'search_verbatim','brain_decay_apply','dreaming_read','dreaming_cycle',
      'learn','active_memory',
    ];
    for (const name of staticTools) {
      registerTool(name, api);
    }
    logger.info('[duckbot-memory] registered %d tools (singleton refs=%d)', _registeredTools.length, _refCount);

    // ---- session hooks ----------------------------------------------------
    // OpenClaw's plugin loader requires every `registerHook(events, handler, opts?)`
    // call to pass `opts.name` (the loader uses `requireRegistrationValue(opts.name,
    // 'hook registration missing name')`). Without it, the entire plugin registration
    // throws and NO tools get registered — every brain tool call would fail with
    // "tool not found". See registry-BXwW-HDh.js: registerHook() validates
    // `entry?.hook.name ?? opts?.name` and throws if both are missing.
    api.registerHook('session_start', fireWakeUp, { name: 'duckbot-memory.session_start' });
    api.registerHook('session_end',   fireSync,   { name: 'duckbot-memory.session_end' });

    // ---- shutdown --------------------------------------------------------
    const gatewayStop = async () => {
      _refCount--;
      logger.info('[duckbot-memory] gateway_stop (refs=%d→%d)', _refCount + 1, _refCount);
      if (_refCount > 0) {
        logger.info('[duckbot-memory] other workers still using singleton — not killing process');
        return;
      }
      // Last worker — shut down the shared process.
      _initiatingShutdown = true;
      logger.info('[duckbot-memory] last worker — shutting down shared Python process (pid=%d)', _sharedChildPid);
      try { _sharedRpc.notify('shutdown', null); } catch { /* gone */ }
      try { _sharedChild.stdin.end(); } catch { /* gone */ }
      try { _sharedChild.kill('SIGTERM'); } catch { /* already dead */ }
      _sharedRpc.close();
      if (_stderrLogStream) {
        try { _stderrLogStream.end(); } catch { /* already closed */ }
      }
      logger.info('[duckbot-memory] singleton shut down');
    };
    api.registerHook('gateway_stop', gatewayStop, { name: 'duckbot-memory.gateway_stop' });

    // ---- diagnostic handle on globalThis -----------------------------------
    globalThis[GLOBAL_KEY] = {
      pid: () => _sharedChildPid,
      repoPath: _repoPath,
      pythonPath: _pythonPath,
      registeredTools: () => [..._registeredTools],
      handshakeTools: () => [..._toolNames],
      refCount: () => _refCount,
      stderrLogPath: _stderrLogPath,
      // Graceful restart: kill child, let child.on('exit') respawn it.
      restart: () => {
        _initiatingShutdown = false;
        logger.info('[duckbot-memory] manual restart — killing child (pid=%d)', _sharedChildPid);
        try { _sharedChild.kill('SIGTERM'); } catch { /* already dead */ }
        _sharedRpc.close();
        // child.on('exit') fires → respawns automatically.
      },
      // Clean shutdown regardless of refcount.
      shutdown: () => {
        _initiatingShutdown = true;
        _refCount = 1;  // force this worker to be the last
        try { _sharedChild.kill('SIGTERM'); } catch { /* already dead */ }
        _sharedRpc.close();
        if (_stderrLogStream) {
          try { _stderrLogStream.end(); } catch { /* already closed */ }
        }
      },
    };
  },
});
