'use strict';

/**
 * JSON-RPC framed I/O over stdio per the MCP spec.
 *
 * Messages are framed as `Content-Length: N\r\n\r\n<json>` (HTTP-style).
 * Some clients send newline-delimited JSON instead — we handle both:
 *   1. Try Content-Length framing first.
 *   2. Fall back to newline-delimited if no header ever appeared.
 *
 * Responses are matched to pending requests by `id`. Each pending request
 * has a default 60s timeout; callers can override per call.
 *
 * Server-initiated notifications (no `id` field) are forwarded to
 * registered listeners via `onMessage(handler)`.
 *
 * Used by `index.js` (the OpenClaw shim entry). Kept in a separate file
 * so unit tests can import + exercise it without booting the full shim.
 */

class StdioJsonRpc {
  constructor(child, logger) {
    this._child = child;
    this._logger = logger;
    this._buffer = Buffer.alloc(0);
    this._pending = new Map();   // id → { resolve, reject, timer }
    this._nextId = 1;
    this._listeners = new Set();
    this._onStdout = this._onStdout.bind(this);
    this._onStderr = this._onStderr.bind(this);
    this._onExit = this._onExit.bind(this);
    child.stdout.on('data', this._onStdout);
    child.stderr.on('data', this._onStderr);
    child.on('exit', this._onExit);
  }

  /** Send a request and await a response (matched by id). */
  send(method, params, timeoutMs = 60_000) {
    const id = this._nextId++;
    const msg = { jsonrpc: '2.0', id, method, params: params || {} };
    const body = Buffer.from(JSON.stringify(msg), 'utf8');
    const framed = Buffer.concat([
      Buffer.from(`Content-Length: ${body.length}\r\n\r\n`, 'ascii'),
      body,
    ]);
    try {
      this._child.stdin.write(framed);
    } catch (e) {
      this._rejectAll(new Error(`stdin closed: ${e.message}`));
      return Promise.reject(e);
    }
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this._pending.delete(id);
        reject(new Error(`timeout waiting for response to ${method} after ${timeoutMs}ms`));
      }, timeoutMs);
      this._pending.set(id, { resolve, reject, timer });
    });
  }

  /** Fire-and-forget (no id, no response expected). */
  notify(method, params) {
    const msg = { jsonrpc: '2.0', method, params: params || {} };
    const body = Buffer.from(JSON.stringify(msg), 'utf8');
    const framed = Buffer.concat([
      Buffer.from(`Content-Length: ${body.length}\r\n\r\n`, 'ascii'),
      body,
    ]);
    try { this._child.stdin.write(framed); } catch { /* process may have exited */ }
  }

  /** Register a handler for server-initiated notifications. */
  onMessage(handler) { this._listeners.add(handler); }

  /** Detach listeners + reject all pending requests. */
  close() {
    this._child.stdout.off('data', this._onStdout);
    this._child.stderr.off('data', this._onStderr);
    this._child.off('exit', this._onExit);
    this._rejectAll(new Error('shim closed'));
  }

  _onStdout(chunk) {
    this._buffer = Buffer.concat([this._buffer, chunk]);
    this._drainBuffer();
  }

  _drainBuffer() {
    // Try Content-Length framing first.
    while (true) {
      const headerEnd = this._buffer.indexOf('\r\n\r\n');
      if (headerEnd === -1) break;
      const header = this._buffer.subarray(0, headerEnd).toString('ascii');
      const m = /Content-Length:\s*(\d+)/i.exec(header);
      if (!m) {
        // Malformed header — drop the buffer to avoid an infinite loop.
        this._buffer = Buffer.alloc(0);
        this._logger.warn('[duckbot-memory] malformed Content-Length header; discarding buffer');
        break;
      }
      const bodyLen = Number(m[1]);
      const totalLen = headerEnd + 4 + bodyLen;
      if (this._buffer.length < totalLen) break; // wait for more data
      const body = this._buffer.subarray(headerEnd + 4, totalLen);
      this._buffer = this._buffer.subarray(totalLen);
      this._dispatchMessage(body);
    }
    // Fallback: if no Content-Length ever appeared, treat each newline
    // as a JSON message (some clients use newline-delimited framing).
    if (this._buffer.length > 0 && !this._buffer.includes('\r\n\r\n') && this._buffer.includes('\n')) {
      const lines = this._buffer.toString('utf8').split('\n').filter(Boolean);
      const consumed = this._buffer.length;
      this._buffer = Buffer.alloc(0);
      let parsedAny = false;
      for (const line of lines) {
        try {
          JSON.parse(line);
          parsedAny = true;
          this._dispatchMessage(Buffer.from(line, 'utf8'));
        } catch { /* keep for restore below */ }
      }
      // If nothing parsed, restore the buffer (it's not JSON-RPC at all).
      if (!parsedAny) {
        this._buffer = Buffer.from(lines.join('\n') + '\n').subarray(0, consumed);
      }
    }
  }

  _dispatchMessage(body) {
    let msg;
    try { msg = JSON.parse(body.toString('utf8')); }
    catch (e) {
      this._logger.warn('[duckbot-memory] non-JSON message: %s', body.toString('utf8').slice(0, 200));
      return;
    }
    if (msg.id != null && this._pending.has(msg.id)) {
      const { resolve, reject, timer } = this._pending.get(msg.id);
      clearTimeout(timer);
      this._pending.delete(msg.id);
      if (msg.error) reject(new Error(msg.error.message || JSON.stringify(msg.error)));
      else resolve(msg.result);
      return;
    }
    // Server-initiated notification — forward to listeners.
    for (const h of this._listeners) {
      try { h(msg); } catch (e) {
        this._logger.debug('[duckbot-memory] listener threw: %s', e.message);
      }
    }
  }

  _onStderr(chunk) {
    const text = chunk.toString('utf8').trim();
    if (text) this._logger.warn('[duckbot-memory] python stderr: %s', text);
  }

  _onExit(code, signal) {
    const reason = code != null ? `exit ${code}` : `signal ${signal}`;
    this._rejectAll(new Error(`Python MCP server terminated (${reason})`));
  }

  _rejectAll(err) {
    for (const { reject, timer } of this._pending.values()) {
      clearTimeout(timer);
      reject(err);
    }
    this._pending.clear();
  }
}

module.exports = { StdioJsonRpc };