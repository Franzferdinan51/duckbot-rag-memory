#!/usr/bin/env python3
"""brain-remember-stdio.py — store a memory in the DuckBot brain via the
local MCP server spoken over stdio with RAW JSON-RPC (no mcp SDK import in
the outer process; avoids cross-venv package resolution issues).

Usage:
  brain-remember-stdio.py "text"                 # one-liner
  brain-remember-stdio.py < file.txt             # text from stdin
  brain-remember-stdio.py --importance 0.9 "..."

Env: builds a clean env for the child server (current env minus PYTHONPATH,
plus the repo .env, plus DUCKBOT_EMBEDDING=lmstudio). Exit 0 = stored.
"""
import json, os, queue, subprocess, sys, threading, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent  # .../duckbot-rag-memory

def build_env() -> dict:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.setdefault("PATH", "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin")
    for line in (REPO / ".env").read_text().splitlines() if (REPO / ".env").exists() else []:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"')
    env["DUCKBOT_EMBEDDING"] = "lmstudio"
    return env

def call(proc, lines, method, params, wid, timeout=90):
    proc.stdin.write((json.dumps({"jsonrpc": "2.0", "id": wid, "method": method, "params": params}) + "\n").encode())
    proc.stdin.flush()
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            raw = lines.get(timeout=0.5)
        except queue.Empty:
            if proc.poll() is not None:
                break
            continue
        raw = raw.strip()
        if not raw:
            continue
        try:
            msg = json.loads(raw)
        except Exception:
            continue
        if msg.get("id") == wid:
            return msg
    return None

def main():
    args = sys.argv[1:]
    importance = 0.9
    if "--importance" in args:
        i = args.index("--importance")
        importance = float(args[i + 1])
        del args[i:i + 2]
    if args:
        text = " ".join(args)
    else:
        text = sys.stdin.read()
    text = text.strip()
    if not text:
        print("no text given", file=sys.stderr)
        sys.exit(2)

    proc = subprocess.Popen(
        [str(REPO / ".venv/bin/python"), "-m", "src.mcp_server"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, env=build_env(), cwd=str(REPO))
    lines = queue.Queue()

    def reader():
        for raw in proc.stdout:
            lines.put(raw.decode("utf-8", "replace"))
    threading.Thread(target=reader, daemon=True).start()

    try:
        res = call(proc, lines, "initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "hermes-stdio", "version": "1.0"}}, 1)
        if not res or "result" not in res:
            print("init failed:", json.dumps(res)[:200] if res else "no response", file=sys.stderr)
            sys.exit(1)
        proc.stdin.write((json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n").encode())
        proc.stdin.flush()
        res = call(proc, lines, "tools/call", {"name": "remember", "arguments": {
            "text": text, "importance": importance,
            "source_path": os.environ.get("HERMES_SOURCE_PATH", "<hermes-stdio>")} }, 2)
        out = json.dumps(res)[:300] if res else "no response"
        ok = res is not None and "result" in res and not res.get("result", {}).get("isError")
        print(("STORED " if ok else "FAILED ") + out)
        sys.exit(0 if ok else 1)
    finally:
        proc.terminate()

if __name__ == "__main__":
    main()
