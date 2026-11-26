#!/bin/bash
# Brain Sync wrapper — activates venv and runs sync with LM Studio embeddings
# Called by the hourly systemd timer (duckbot-brain-sync).
#
# Canonical copy of ~/rag-jobs/brain-sync-wrapper.sh on Robin.
# Keep the two in sync; the Robin copy is what the timer runs.
#
# v0.18.0: target is HERMES ONLY. Per the owner's 2026-11-23 ruling the
# brain serves Hermes + Juno exclusively; the old combined target also
# wrote OpenClaw context files every hour, feeding a system the boundary
# excludes. The openclaw target still exists in the CLI for explicit,
# owner-invoked use — it is just no longer on a timer.
cd /home/duckets/duckbot-rag-memory
. .venv/bin/activate
export DUCKBOT_EMBEDDING=lmstudio
unset PYTHONPATH
# Self-heal: the sync needs the embed model loaded; load it if absent (84 MB).
lms ps 2>/dev/null | grep -q "nomic-embed" || lms load text-embedding-nomic-embed-text-v1.5 2>/dev/null
python -m src.cli sync --target hermes

# Cleanup: unload the embedder we self-loaded (frees ~84 MB; next run self-heals).
lms unload text-embedding-nomic-embed-text-v1.5 >/dev/null 2>&1 || true
