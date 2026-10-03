#!/usr/bin/env python3
"""Cron-safe probe: brain_stats + brain_block_list (read-only)."""
import sys, os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))
os.chdir(REPO_ROOT)

from dotenv import load_dotenv
load_dotenv('.env')

from src.memory import Memory
from src.blocks import BlockStore

mem = Memory()
blocks_store = BlockStore()

print("=== brain_stats ===")
try:
    stats = mem.stats()
    print(stats)
except Exception as e:
    print(f"ERROR brain_stats: {e}")

print("\n=== brain_block_list ===")
try:
    blocks = blocks_store.list_blocks()
    for b in blocks:
        print(f"  {b.name} ({len(b.content)} chars)")
except Exception as e:
    print(f"ERROR brain_block_list: {e}")
