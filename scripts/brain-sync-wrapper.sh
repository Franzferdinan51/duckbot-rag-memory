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
# excludes.
# v0.19.0: OpenClaw targets are gone from the CLI entirely (not just off
# the timer). The embedder self-heal now runs over the LM Studio REST
# API — Robin has no `lms` binary, so the old lms lines were silent
# no-ops — and the embedder is NEVER unloaded afterwards: a cold
# embedder makes brain_wake_up take ~8.6s, over Hermes' 8s prefetch
# timeout (the 2026-11-26 "Running brain wake up" hang).
cd /home/duckets/duckbot-rag-memory
. .venv/bin/activate
export DUCKBOT_EMBEDDING=lmstudio
unset PYTHONPATH

# Self-heal: ensure the embed model is loaded in LM Studio (84 MB).
python3 - << 'HEAL_EOF'
import json, os, urllib.request

def _lm_base() -> str:
    url = os.environ.get("LMSTUDIO_URL", "")
    if not url:
        try:
            for line in open(".env"):
                if line.startswith("LMSTUDIO_URL="):
                    url = line.split("=", 1)[1].strip()
                    break
        except OSError:
            pass
    url = url or "http://100.68.208.113:1234/v1"
    return url[:-3] if url.endswith("/v1") else url

MODEL = "text-embedding-nomic-embed-text-v1.5"
try:
    base = _lm_base()
    with urllib.request.urlopen(base + "/api/v1/models", timeout=10) as r:
        models = json.load(r).get("models", [])
    entry = next((m for m in models if m.get("key") == MODEL), None)
    if entry is not None and not entry.get("loaded_instances"):
        req = urllib.request.Request(
            base + "/api/v1/models/load",
            data=json.dumps({"model": MODEL}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=120).read()
        print(f"embedder self-heal: loaded {MODEL}")
except Exception as e:  # never fail the sync over a heal attempt
    print(f"embedder self-heal skipped: {e}")
HEAL_EOF

python -m src.cli sync --target hermes
