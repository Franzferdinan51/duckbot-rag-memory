# Cron Entries to Add

Add these to your user crontab (run `crontab -e` or `crontab /path/to/this/file`):

```
# duckbot-rag-memory: nightly full-state backup at 3am
0 3 * * * /Users/duckets/Desktop/duckbot-rag-memory/scripts/backup-brain.sh >> /Users/duckets/Desktop/duckbot-rag-memory/data/backup-cron.log 2>&1

# duckbot-rag-memory: prune backups older than 14 days at 3:30am
30 3 * * * find /Users/duckets/Desktop/duckbot-rag-memory/data/backups -name 'brain-backup-*.tar.gz' -mtime +14 -delete >> /Users/duckets/Desktop/duckbot-rag-memory/data/backup-cron.log 2>&1
```

## Why these times?
- **3am**: low-activity window on your machine
- **3:30am prune**: 30min after backup so today's backup is preserved
- **14-day retention**: enough history to recover from any recent wipe

## Manual backup
```bash
/Users/duckets/Desktop/duckbot-rag-memory/scripts/backup-brain.sh
```

## Restore from backup
```bash
/Users/duckets/Desktop/duckbot-rag-memory/scripts/restore-brain.sh --dry-run  # verify first
/Users/duckets/Desktop/duckbot-rag-memory/scripts/restore-brain.sh             # actually restore latest
```
