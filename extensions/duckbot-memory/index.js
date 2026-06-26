/**
 * DuckBot Memory — OpenClaw native plugin (Node.js shim).
 *
 * Pure Node.js, zero npm dependencies. Spawns the Python MCP server
 * (duckbot-rag-memory/src/mcp_server.py) as a subprocess and proxies
 * 64 brain tools + session_start / session_end hooks into OpenClaw.
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
 * bridge the 64 tools via JSON-RPC. No code duplication — the Python
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

    logger.info(
      '[duckbot-memory] starting Python MCP server (repo=%s, python=%s)',
      repoPath, pythonPath,
    );

    // ---- v0.15.1: tee Python stderr to data/mcp.log ---------------------
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

    // ---- spawn Python MCP server as a subprocess --------------------------
    const child = spawn(
      pythonPath,
      ['-u', '-m', 'src.mcp_server'],
      {
        cwd: repoPath,
        env: {
          ...process.env,
          PYTHONUNBUFFERED: '1',
          // Don't let the brain's own .env-loading compete with ours.
          DUCKBOT_EMBEDDING: process.env.DUCKBOT_EMBEDDING || 'lmstudio',
          DUCKBOT_REPO_PATH: repoPath,
        },
        stdio: ['pipe', 'pipe', 'pipe'],
      },
    );

    // Tee Python stderr to the log file (in addition to the plugin logger,
    // which StdioJsonRpc._onStderr already handles).
    if (stderrLogStream) {
      child.stderr.on('data', (chunk) => {
        stderrLogStream.write(chunk);
      });
    }

    const rpc = new StdioJsonRpc(child, logger);
    const handle = { child, rpc, closed: false, stderrLogStream, stderrLogPath };

    child.on('error', (err) => {
      logger.error('[duckbot-memory] failed to spawn python: %s', err.message);
    });

    // ---- initialize handshake ---------------------------------------------
    let toolNames = [];
    let initialized = false;
    (async () => {
      try {
        await rpc.send('initialize', {
          protocolVersion: '2024-11-05',
          capabilities: {},
          clientInfo: { name: 'duckbot-memory-openclaw-shim', version: '0.1.0' },
        });
        rpc.notify('notifications/initialized', {});
        // List tools so we know what to register.
        const { tools } = await rpc.send('tools/list', {});
        toolNames = tools.map((t) => t.name);
        initialized = true;
        logger.info('[duckbot-memory] ready: %d tools registered', toolNames.length);
      } catch (e) {
        logger.error('[duckbot-memory] MCP initialize failed: %s', e.message);
      }
    })();

    // ---- factory: returns an AnyAgentTool whose execute() proxies a call --
    function makeToolFactory(toolName) {
      return () => ({
        name: toolName,
        description: `DuckBot brain tool: ${toolName}`,
        parameters: { type: 'object', properties: {}, additionalProperties: true },
        async execute(args) {
          if (!initialized) {
            throw new Error('duckbot-memory shim still initializing; retry in a moment');
          }
          // Enforce per-call timeout. The Python side has its own ceiling
          // for individual ops (eval/reflect can take seconds); this is
          // the outer envelope so a stuck call doesn't hang the agent.
          const controller = new AbortController();
          const timer = setTimeout(() => controller.abort(), timeoutMs);
          try {
            const result = await rpc.send('tools/call', {
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

    // Register every tool the Python server reported. We register them
    // eagerly (without waiting for initialize) so OpenClaw advertises the
    // surface immediately; the factory's execute() will surface a clear
    // "still initializing" error if the agent calls before we're ready.
    const registeredTools = [];
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
      'brain_export','brain_import','brain_seed_demo',
      'remember','recall','reflect','forget','stats','watch','doctor',
      'recall_verbatim','fsrs_review','decay_status','forget_by_query',
      'search_verbatim','brain_decay_apply','dreaming_read','dreaming_cycle',
      'learn','active_memory',
    ]) {
      try {
        api.registerTool(makeToolFactory(name), { name });
        registeredTools.push(name);
      } catch (e) {
        // OpenClaw may reject duplicate names or unsupported shapes;
        // log and continue.
        logger.debug('[duckbot-memory] registerTool(%s) skipped: %s', name, e.message);
      }
    }
    logger.info('[duckbot-memory] registered %d tools (handshake reports %d)', registeredTools.length, toolNames.length);

    // ---- session_start hook: auto-fire brain_wake_up ----------------------
    api.registerHook('session_start', async (event, ctx) => {
      if (!autoWakeUp) return;
      if (!initialized) return; // shim not ready yet; will be picked up next turn
      try {
        const wake = await rpc.send('tools/call', {
          name: 'brain_wake_up',
          arguments: { k: defaultK, include_blocks: true, include_graph: true, include_fsrs_review: true },
        });
        // The MCP result is { content: [{type: 'text', text: '<json>'}], isError? }.
        // Surface it as a system-prompt contribution if the runtime supports it.
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
      if (handle.closed) return;
      handle.closed = true;
      try { rpc.notify('shutdown', null); } catch { /* already gone */ }
      try { child.stdin.end(); } catch { /* already gone */ }
      try { child.kill('SIGTERM'); } catch { /* already gone */ }
      rpc.close();
      // Flush the stderr log stream so the tail of the Python output
      // lands on disk before the gateway tears us down.
      if (handle.stderrLogStream) {
        try { handle.stderrLogStream.end(); } catch { /* already closed */ }
      }
      logger.info('[duckbot-memory] shim shut down');
    });

    // ---- expose a handle on globalThis for diagnostics ---------------------
    globalThis[GLOBAL_KEY] = {
      pid: child.pid,
      repoPath,
      pythonPath,
      registeredTools: () => [...registeredTools],
      handshakeTools: () => [...toolNames],
      stderrLogPath,
      shutdown: () => {
        try { child.kill('SIGTERM'); } catch { /* already gone */ }
        rpc.close();
        if (handle.stderrLogStream) {
          try { handle.stderrLogStream.end(); } catch { /* already closed */ }
        }
      },
    };
  },
});