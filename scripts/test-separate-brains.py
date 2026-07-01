#!/usr/bin/env python3
"""Test both brains with separate data dirs via duck-memory launcher."""
import subprocess, os, json

REPO = r"C:\Users\franz\Desktop\duckbot-rag-memory"
LAUNCHER = os.path.join(REPO, "duck-memory")

def run(args, extra_env=None):
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", **(extra_env or {})}
    r = subprocess.run(
        [LAUNCHER] + args,
        capture_output=True, text=True, cwd=REPO, env=env, timeout=30
    )
    return r.stdout + r.stderr

# Test duck-memory --data-dir (if supported) or env-based isolation
print("=== duck-memory doctor ===")
print(run(["doctor"])[:400])
print()

print("=== duck-memory stats ===")
print(run(["stats"])[:300])
