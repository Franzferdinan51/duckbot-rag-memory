/**
 * DuckBot Memory — OpenClaw native plugin (Node.js shim).
 *
 * Pure Node.js, zero npm dependencies. Spawns the Python MCP server
 * (duckbot-rag-memory/src/mcp_server.py) as a subprocess and proxies
 * 68 brain tools + session_start / session_end hooks into OpenClaw.
 *
 * Pattern sources:
 *   - openclaw/openclaw `extensions/voice-call/index.ts` (definePluginEntry)
 *   - openclaw/openclaw `docs/plugins/manifest.md` (openclaw.plugin.json)
 *   - MCP spec (Content-Length framed JSON-RPC over stdio)
 *     https://spec.modelcontextprotocol.io/specification/basic/transports/
 *
 * Why a shim and not a native Python plugin: OpenClaw plugins run in-process
 * inside the Node gateway (src/plugins/loader.ts). Python isn't supported
 * natively. So we spawn the existing Python MCP server as a subprocess and
 * bridge the 68 tools via JSON-RPC. No code duplication — the Python
 * `src/mcp_server.py` IS the brain.
 */

'use strict';

const { spawn } = require('node:child_process');
const { existsSync } = require('node:fs');
const path = require('node:path');
const process = require('node:process');

const GLOBAL_KEY = Symbol.for('openclaw.duckbot-memory');
const { StdioJsonRpc } = require('./lib/stdio-rpc');

/**
 * Resolve the default `pythonPath` for the repo's venv if the user didn't
 * override it via plugin config.
 */
function defaultPythonPath(repoPath) {
  if (process.platform === 'win32') {
    const p = path.join(repoPath, '.venv', 'Scripts', 'python.exe');
    return existsSync(p) ? p : 'python';
  }
  const p = path.join(repoPath, '.venv', 'bin', 'python');
  return existsSync(p) ? p : 'python3';
}

/**
 * Plugin entry — OpenClaw loads this as the default export.
 * Pattern: extensions/voice-call/index.ts → `definePluginEntry({...})`.
 */
function definePluginEntry(opts) { return opts; }

module.exports = definePluginEntry({
  id: 'duckbot-memory',
  configSchema: require('./openclaw.plugin.json').configSchema,

  register(api) {
    const logger = api.logger;
    const cfg = api.pluginConfig || {};

    // ---- resolve config (with sensible defaults) --------------------------
    const repoPath = cfg.repoPath || process.env.DUCKBOT_REPO_PATH;
    if (!repoPath) {
      logger.error('[duckbot-memory] repoPath is required (set in plugin config or DUCKBOT_REPO_PATH env)');
      return;
    }
    const pythonPath = cfg.pythonPath || process.env.DUCKBOT_PYTHON_PATH || defaultPythonPath(repoPath);
    const defaultK = Number.isFinite(cfg.defaultK) ? cfg.defaultK : 5;
    const autoWakeUp = cfg.autoWakeUp !== false;
    const autoSync = cfg.autoSync !== false;
    const timeoutMs = Number.isFinite(cfg.timeoutMs) ? cfg.timeoutMs : 15_000;

    // ---- stderr log: tee Python stderr to data/mcp.log --------------------
    // stdout is the JSON-RPC channel (OpenClaw reads it line-by-line) so
    // we must NOT redirect stdout. stderr is free — operators can read
    // data/mcp.log to debug segfaults and tracebacks after the fact.
    // Set DUCKBOT_MCP_LOG=/path/to/file (or default data/mcp.log) to
    // override the destination; set to empty string to disable.
    let stderrLogStream = null;
    let stderrLogPath = null;
    const logPathEnv = process.env.DUCKBOT_MCP_LOG;
    if (logPathEnv !== '') {
      stderrLogPath = logPathEnv || path.join(repoPath, 'data', 'mcp.log');
      try {
        const fs = require('node:fs');
        fs.mkdirSync(path.dirname(stderrLogPath), { recursive: true });
        stderrLogStream = fs.createWriteStream(stderrLogPath, { flags: 'a' });
        stderrLogStream.on('error', (e) => {
          logger.warn('[duckbot-memory] stderr log stream error: %s', e.message);
        });
      } catch (e) {
        logger.warn('[duckbot-memory] could not open %s: %s — stderr will go to logger only', stderrLogPath, e.message);
        stderrLogStream = null;
      }
    }

    // ---- state shared across spawn / respawn cycles -------------------------
    let initiatingShutdown = false;
    let childPid = null;
    let rpc = null;  // current active RPC connection

    // ---- tool registration state (rebuilt on each respawn) -----------------
    let toolNames = [];
    let initialized = false;
    const registeredTools = [];
    const registeredToolSet = new Set();

    // ---- spawn a fresh Python MCP server child process --------------------
    function spawnChild() {
      const child = spawn(
        pythonPath,
        ['-u', '-m', 'src.mcp_server'],
        {
          cwd: repoPath,
          env: {
            ...process.env,
            PYTHONUNBUFFERED: '1',
            DUCKBOT_EMBEDDING: process.env.DUCKBOT_EMBEDDING || 'lmstudio',
            DUCKBOT_REPO_PATH: repoPath,
          },
          stdio: ['pipe', 'pipe', 'pipe'],
        },
      );
      childPid = child.pid;

      // Tee Python stderr to the log file (skip urllib3 NotOpenSSLWarning noise).
      if (stderrLogStream) {
        child.stderr.on('data', (chunk) => {
          const text = chunk.toString('utf8');
          // Suppress the LibreSSL/OpenSSL version mismatch warnings — these are
          // cosmetic and don't affect functionality.
          if (text.includes('NotOpenSSLWarning')) return;
          stderrLogStream.write(chunk);
        });
      }

      child.on('error', (err) => {
        logger.error('[duckbot-memory] python child error (pid=%d): %s', childPid, err.message);
      });

      // Auto-respawn when the Python process exits (unless we're shutting down).
      child.on('exit', (code, signal) => {
        const reason = code != null ? `exit ${code}` : `signal ${signal}`;
        logger.info('[duckbot-memory] Python MCP server exited (%s, pid=%d)', reason, childPid);
        if (initiatingShutdown) {
          logger.info('[duckbot-memory] intentional shutdown — not respawning');
          return;
        }
        // Respawn after a short delay so any port / file handle is released.
        logger.info('[duckbot-memory] auto-respawning Python MCP server in 2s...');
        setTimeout(() => { spawnChild(); }, 2000);
      });

      return child;
    }

    // ---- create the initial child process ------------------------------------
    const child = spawnChild();
    rpc = new StdioJsonRpc(child, logger);
    const handle = { child, rpc, closed: false, stderrLogStream, stderrLogPath };

    // ---- tool factory: closes over the current rpc instance -----------------
    function makeToolFactory(toolName, rpcInstance) {
      const activeRpc = rpcInstance || rpc;
      return () => ({
        name: toolName,
        description: `DuckBot brain tool: ${toolName}`,
        parameters: { type: 'object', properties: {}, additionalProperties: true },
        async execute(args) {
          if (!initialized) {
            throw new Error('duckbot-memory shim still initializing; retry in a moment');
          }
          const controller = new AbortController();
          const timer = setTimeout(() => controller.abort(), timeoutMs);
          try {
            const result = await activeRpc.send('tools/call', {
              name: toolName,
              arguments: args || {},
            });
            return result;
          } catch (e) {
            if (e.name === 'AbortError') {
              throw new Error(`duckbot-memory tool '${toolName}' timed out after ${timeoutMs}ms`);
            }
            throw e;
          } finally {
            clearTimeout(timer);
          }
        },
      });
    }

    function registerToolName(name, rpcInstance) {
      if (registeredToolSet.has(name)) return false;
      try {
        api.registerTool(makeToolFactory(name, rpcInstance), { name });
        registeredTools.push(name);
        registeredToolSet.add(name);
        return true;
      } catch (e) {
        logger.debug('[duckbot-memory] registerTool(%s) skipped: %s', name, e.message);
        return false;
      }
    }

    // ---- re-initialize after a respawn ------------------------------------
    async function reinitialize(newRpc, newChild) {
      try {
        await newRpc.send('initialize', {
          protocolVersion: '2024-11-05',
          capabilities: {},
          clientInfo: { name: 'duckbot-memory-openclaw-shim', version: '0.1.0' },
        });
        newRpc.notify('notifications/initialized', {});
        const { tools } = await newRpc.send('tools/list', {});
        const newNames = tools.map((t) => t.name);
        let newlyRegistered = 0;
        for (const name of newNames) {
          if (registerToolName(name, newRpc)) newlyRegistered += 1;
        }
        initialized = true;
        // Update the handle so shutdown() kills the current child.
        handle.child = newChild;
        handle.rpc = newRpc;
        logger.info(
          '[duckbot-memory] respawn complete: %d tools discovered, %d newly registered (total=%d)',
          newNames.length, newlyRegistered, registeredTools.length,
        );
      } catch (e) {
        logger.error('[duckbot-memory] re-initialize failed: %s — plugin may need gateway restart', e.message);
      }
    }

    // ---- initial MCP handshake ---------------------------------------------
    (async () => {
      try {
        await rpc.send('initialize', {
          protocolVersion: '2024-11-05',
          capabilities: {},
          clientInfo: { name: 'duckbot-memory-openclaw-shim', version: '0.1.0' },
        });
        rpc.notify('notifications/initialized', {});
        const { tools } = await rpc.send('tools/list', {});
        toolNames = tools.map((t) => t.name);
        let newlyRegistered = 0;
        for (const name of toolNames) {
          if (registerToolName(name)) newlyRegistered += 1;
        }
        initialized = true;
        logger.info(
          '[duckbot-memory] ready: %d tools discovered, %d newly registered, %d total registered (pid=%d)',
          toolNames.length, newlyRegistered, registeredTools.length, childPid,
        );
      } catch (e) {
        logger.error('[duckbot-memory] MCP initialize failed: %s', e.message);
      }
    })();

    // Register every tool the Python server reported. We register them
    // eagerly (without waiting for initialize) so OpenClaw advertises the
    // surface immediately; the factory's execute() will surface a clear
    // "still initializing" error if the agent calls before we're ready.
    for (const name of [
      // Canonical MCP server tools — listed statically so OpenClaw's
      // tools/list discovery sees them before our initialize handshake
      // completes. Dynamic discovery (above) keeps this in sync at runtime.
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
    ]) {
      registerToolName(name);
    }
    logger.info('[duckbot-memory] registered %d tools (handshake reports %d)', registeredTools.length, toolNames.length);

    // ---- session_start hook: auto-fire brain_wake_up ----------------------
    api.registerHook('session_start', async (event) => {
      if (!autoWakeUp) return;
      if (!initialized) return; // shim not ready yet; will be picked up next turn
      try {
        const wake = await rpc.send('tools/call', {
          name: 'brain_wake_up',
          arguments: { k: defaultK, include_blocks: true, include_graph: true, include_fsrs_review: true },
        });
        const text = wake?.content?.[0]?.text || '';
        if (text && api.runtime?.llm?.injectSystemPrompt) {
          api.runtime.llm.injectSystemPrompt(
            `\n[duckbot-memory: session_start brain_wake_up]\n${text}\n`,
          );
        }
        logger.info('[duckbot-memory] session_start: brain_wake_up fired (session=%s)', event?.sessionId);
      } catch (e) {
        logger.warn('[duckbot-memory] session_start hook failed: %s', e.message);
      }
    });

    // ---- session_end hook: auto-fire brain_sync ----------------------------
    api.registerHook('session_end', async (event) => {
      if (!autoSync) return;
      if (!initialized) return;
      try {
        await rpc.send('tools/call', {
          name: 'brain_sync',
          arguments: { target: 'openclaw', dry_run: false },
        });
        logger.info('[duckbot-memory] session_end: brain_sync fired (session=%s, msgs=%d)',
                    event?.sessionId, event?.messageCount);
      } catch (e) {
        logger.warn('[duckbot-memory] session_end hook failed: %s', e.message);
      }
    });

    // ---- shutdown: clean exit ---------------------------------------------
    api.registerHook('gateway_stop', async () => {
      initiatingShutdown = true;  // prevent auto-respawn during intentional shutdown
      if (handle.closed) return;
      handle.closed = true;
      try { rpc.notify('shutdown', null); } catch { /* already gone */ }
      try { child.stdin.end(); } catch { /* already gone */ }
      try { child.kill('SIGTERM'); } catch { /* already gone */ }
      rpc.close();
      if (handle.stderrLogStream) {
        try { handle.stderrLogStream.end(); } catch { /* already closed */ }
      }
      logger.info('[duckbot-memory] shim shut down');
    });

    // ---- expose a handle on globalThis for diagnostics ---------------------
    globalThis[GLOBAL_KEY] = {
      pid: () => childPid,
      repoPath,
      pythonPath,
      registeredTools: () => [...registeredTools],
      handshakeTools: () => [...toolNames],
      stderrLogPath,
      // Graceful shutdown (intentional — no respawn).
      shutdown: () => {
        initiatingShutdown = true;
        try { child.kill('SIGTERM'); } catch { /* already gone */ }
        rpc.close();
        if (handle.stderrLogStream) {
          try { handle.stderrLogStream.end(); } catch { /* already closed */ }
        }
      },
      // Graceful restart: kill child, let child.on('exit') auto-respawn.
      restart: () => {
        initiatingShutdown = false;
        logger.info('[duckbot-memory] manual restart — killing child (pid=%d)', childPid);
        try { child.kill('SIGTERM'); } catch { /* already gone */ }
        // child.on('exit') will fire and call spawnChild() → reinitialize() automatically.
      },
    };
  },
});
