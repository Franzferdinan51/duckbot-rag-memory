# DuckBot-RAG-Memory Backup Manifest

Generated: 2026-06-30T17:43:49-04:00
Hostname:  Ryans-Mac-mini.local
Brain dir: /Users/duckets/Desktop/duckbot-rag-memory
Archive:   /Users/duckets/Desktop/duckbot-rag-memory/data/backups/brain-backup-2026-06-30_174348.tar.gz

## Contents

- brain_export.md (   67918 lines)
- data/chroma/ (ChromaDB HNSW indexes + SQLite metadata)
- data/blocks.db, graph.db, events.db (SQLite metadata)
- data/watcher_state.json (file hashes + skip flags)
- data/ingest_history.jsonl, eval_history.jsonl
- scripts/, src/ (so the backup is self-describing)
- .env (secrets — keep this archive private)

## Sizes

 86M	/Users/duckets/Desktop/duckbot-rag-memory/data/chroma
3.2M	/Users/duckets/Desktop/duckbot-rag-memory/data/brain_export.md
4.0K	/Users/duckets/Desktop/duckbot-rag-memory/data/backups

## Restore procedure

```bash
launchctl unload ~/Library/LaunchAgents/com.duckbot.memory-watcher.plist
tar -xzf brain-backup-2026-06-30_174348.tar.gz -C ~/Desktop/
cd ~/Desktop/duckbot-rag-memory
.venv/bin/python -m src.cli import --in-path data/brain_export.md
launchctl load ~/Library/LaunchAgents/com.duckbot.memory-watcher.plist
```
