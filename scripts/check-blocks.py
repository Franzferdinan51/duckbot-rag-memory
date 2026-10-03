#!/usr/bin/env python3
import os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))
from dotenv import load_dotenv; load_dotenv('.env')
import sqlite3
conn = sqlite3.connect(str(REPO_ROOT / 'data' / 'blocks.db'))
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print('tables:', cur.fetchall())
cur.execute('SELECT COUNT(*) FROM blocks')
print('block rows:', cur.fetchone())
