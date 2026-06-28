#!/bin/bash
cd /Users/duckets/Desktop/duckbot-rag-memory
nohup .venv/bin/python -m src.watcher run \
    /Users/duckets/.openclaw/workspace/memory \
    /Users/duckets/.openclaw/workspace/SOUL.md \
    /Users/duckets/.openclaw/workspace/MEMORY.md \
    /Users/duckets/.openclaw/workspace/USER.md \
    /Users/duckets/.openclaw/workspace/AGENTS.md \
    >> data/watcher.log 2>&1 &
