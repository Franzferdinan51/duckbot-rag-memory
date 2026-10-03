#!/usr/bin/env python3
"""Cron-run brain decay prune for episodic tier."""
import asyncio, signal, sys, json, os

def _t(sig, frame):
    print("TIMEOUT: decay exceeded 90s", file=sys.stderr)
    sys.exit(124)

signal.signal(signal.SIGALRM, _t)
signal.alarm(90)

# Ensure repo root on path
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("DUCKBOT_EMBEDDING", "local")

from src.mcp_server import handle_brain_decay_apply

async def main():
    result = await handle_brain_decay_apply({
        "dry_run": False,
        "retention_floor": 0.2,
        "tier": "episodic",
    })
    print(json.dumps(result, indent=2, default=str))

asyncio.run(main())
