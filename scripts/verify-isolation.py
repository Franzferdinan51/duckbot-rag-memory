#!/usr/bin/env python3
"""Verify DuckMemory MCP data isolation — DuckMemory env var."""
import subprocess, os, json

REPO = r"C:\Users\franz\Desktop\duckbot-rag-memory"
DuckMemory = os.path.join(REPO, ".venv", "Scripts", "python.exe")

def mcp_stats(env_overrides):
    inp = json.dumps({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "stats", "arguments": {}}})
    env = dict(os.environ)
    env.update(env_overrides)
    env["DuckMemory"] = DuckMemory
    env["PYTHONIOENCODING"] = "utf-8"
    r = subprocess.run(
        [DuckMemory, "-m", "src.mcp_server"],
        input=inp,
        capture_output=True, text=True,
        cwd=REPO, env=env, timeout=20,
    )
    try:
        d = json.loads(r.stdout)
        return d["result"]["content"][0]["text"]
    except Exception:
        return (r.stderr or r.stdout)[:300]

print("=== DuckMemory OpenClaw (DuckMemory=~/.duck-memory) ===")
ocw = mcp_stats({"DuckMemory": os.path.expanduser("~/.duck-memory")})
print(ocw[:300])

print("\n=== DuckMemory Hermès (DuckMemory=~/.duckbot-hermes) ===")
hermes = mcp_stats({"DuckMemory": os.path.expanduser("~/.duckbot-hermes")})
print(hermes[:300])

print("\n=== DuckMemory default ===")
dflt = mcp_stats({})
print(dflt[:300])
