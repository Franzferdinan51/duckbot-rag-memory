#!/usr/bin/env python3
"""Probe MCP stdio server: brain_stats + brain_block_list via subprocess."""
import subprocess, json, sys, os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
script = str(REPO_ROOT / "scripts" / "duckbot-memory-mcp.sh")

req1 = json.dumps({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"brain_stats","arguments":{}}})
req2 = json.dumps({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"brain_block_list","arguments":{}}})

proc = subprocess.Popen(
    ["bash", script],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd=str(REPO_ROOT),
    text=True,
    env=dict(os.environ, PYTHONPATH=str(REPO_ROOT)),
)

out, err = proc.communicate(input=req1 + "\n" + req2 + "\n", timeout=30)

print("=== STDOUT ===")
print(out[:3000])
print("=== STDERR ===")
print(err[:1000])
