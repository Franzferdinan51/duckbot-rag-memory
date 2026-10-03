#!/usr/bin/env python3
"""Probe DuckBot RAG memory: brain_stats + brain_block_list (read-only)."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/duckbot-rag-memory/src"))

# Load .env
from pathlib import Path
env_path = Path("~/duckbot-rag-memory/.env").expanduser()
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)

PYTHONPATH = os.path.expanduser("~/duckbot-rag-memory")
sys.path.insert(0, PYTHONPATH)

from src.memory import Memory
from src.blocks import list_blocks

mem = Memory()

print("=== brain_stats ===")
try:
    stats = mem.stats()
    print(stats)
except Exception as e:
    print(f"ERROR brain_stats: {e}")

print("\n=== brain_block_list ===")
try:
    blocks = list_blocks()
    print(blocks)
except Exception as e:
    print(f"ERROR brain_block_list: {e}")
