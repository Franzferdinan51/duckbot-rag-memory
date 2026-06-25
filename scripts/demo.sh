#!/usr/bin/env bash
# scripts/demo.sh — one-shot end-to-end demo.
#
# Seeds the brain with the bundled demo corpus, then runs the full
# wake_up → recall → answer loop so a new user can see the whole
# pipeline in <30 seconds. Idempotent: re-running skips chunks that
# already exist.
#
# Usage:
#   ./scripts/demo.sh
#
# Requires: venv already created (./scripts/install.sh) and
# DUCKBOT_EMBEDDING configured in .env.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ -f "$REPO_ROOT/.venv/bin/python" ]; then
    PY="$REPO_ROOT/.venv/bin/python"
elif [ -f "$REPO_ROOT/.venv/Scripts/python.exe" ]; then
    PY="$REPO_ROOT/.venv/Scripts/python.exe"
else
    echo "❌ No venv found at $REPO_ROOT/.venv/{bin,Scripts}/python" >&2
    echo "   Run ./scripts/install.sh first." >&2
    exit 1
fi

echo "🧠 DuckBot brain — end-to-end demo"
echo "================================="
echo

# 1. Doctor — confirm setup
echo "→ Step 1/4: doctor (confirm setup)..."
"$PY" -m src.cli doctor 2>&1 | head -20
echo

# 2. Seed the demo corpus
echo "→ Step 2/4: seed the demo corpus (idempotent)..."
"$PY" -m src.cli seed-demo 2>&1 | tail -5
echo

# 3. brain_wake_up — the one-call session-start context load
echo "→ Step 3/4: brain_wake_up (session-start context)..."
"$PY" -m src.cli wake-up 2>&1 | head -40
echo
echo "   ...(truncated; use --json for the full output)..."
echo

# 4. brain_recall — the actual question
echo "→ Step 4/4: brain_recall (sample question)..."
echo "    question: \"How do I restart the BATMAN container?\""
echo
"$PY" -m src.cli query "How do I restart the BATMAN container?" -n 3 2>&1
echo

echo "✓ Demo complete."
echo
echo "Next steps:"
echo "  - ./scripts/openclaw-bootstrap.sh   (ingest your full OpenClaw workspace)"
echo "  - ./scripts/hermes-bootstrap.sh      (ingest your full Hermes workspace)"
echo "  - python -m src.cli wake-up --json   (machine-readable wake_up output)"
echo "  - python -m src.cli palace            (Wing/Room/Drawer view)"
echo "  - python -m src.cli nudge             (proactive memory nudges)"
echo "  - See INSTALL.md for the full install recipe."