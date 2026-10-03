#!/usr/bin/env python3
"""Cron-safe probe: brain_stats + brain_block_list via stdio MCP."""
import sys, json, subprocess, os, time

REPO = os.path.expanduser("~/duckbot-rag-memory")
PYBIN = os.path.join(REPO, ".venv/bin/python")

proc = subprocess.Popen(
    [PYBIN, "-m", "src.mcp_server"],
    cwd=REPO,
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
)
out_lines = []

def drain_stdout():
    try:
        for line in proc.stdout:
            out_lines.append(line.decode())
    except Exception:
        pass

import threading
t = threading.Thread(target=drain_stdout)
t.start()

send = lambda req: (
    proc.stdin.write((json.dumps(req) + "\n").encode()),
    proc.stdin.flush()
)

# Initialize
send({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","clientInfo":{"name":"cron-probe","version":"1.0"},"_meta":{}}})
send({"jsonrpc":"2.0","method":"notifications/initialized","params":{}})

# Two read-only probes
send({"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"brain_stats","arguments":{}}})
send({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"brain_block_list","arguments":{}}})

time.sleep(10)
proc.stdin.close()
proc.wait(timeout=5)
t.join(timeout=3)

stderr = b"".join(proc.stderr.readlines()).decode("utf8", errors="replace")

print("=== STDOUT ===")
for l in out_lines:
    print(l.rstrip())
print("=== STDERR (last 1000) ===")
print(stderr[-1000:] if stderr else "(none)")
