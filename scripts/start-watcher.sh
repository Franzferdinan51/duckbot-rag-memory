#!/usr/bin/env bash
# Start the watcher fully detached from the calling shell.
set -e
cd /Users/duckets/Desktop/duckbot-rag-memory
rm -f data/watcher.pid
# Use nohup + setsid-like behavior by closing all fds and using & + disown
nohup ./.venv/bin/python -m src.watcher run </dev/null >/tmp/watcher-out.log 2>&1 &
WPID=$!
disown $WPID 2>/dev/null
echo "Spawned pid=$WPID"
# Don't sleep here — the parent script will exit and the watcher must survive.
# The watcher writes its own pidfile in cmd_run; check after a moment.
sleep 1
exit 0