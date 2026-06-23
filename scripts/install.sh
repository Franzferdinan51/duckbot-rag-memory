#!/usr/bin/env bash
# scripts/install.sh — bootstrap the project + wire the cron.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=== DuckBot RAG + Memory install ==="
echo "Repo: $REPO_ROOT"

# 1. venv
if [[ ! -d .venv ]]; then
  echo "Creating venv..."
  python3 -m venv .venv
fi

# 2. install deps
echo "Installing deps..."
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet -r requirements.txt

# 3. .env
if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from template. EDIT IT to set OPENAI_API_KEY."
else
  echo ".env already exists"
fi

# 4. make scripts executable
chmod +x scripts/*.sh

# 5. git init if needed
if [[ ! -d .git ]]; then
  git init
  git add -A
  git commit -m "init: duckbot-rag-memory v0.1.0"
  echo "Initialized git repo. Push with: git remote add origin <url> && git push -u origin main"
else
  echo "Git repo already initialized"
fi

# 6. Wire the cron (macOS launchd or Linux cron)
echo ""
echo "Cron schedule (12 invocations across 22:00-10:00):"
echo "  0 22-23,0-9 * * *  bash $REPO_ROOT/scripts/cron.sh"
echo ""
read -p "Wire this into your crontab now? [y/N] " install_cron
if [[ "$install_cron" == "y" || "$install_cron" == "Y" ]]; then
  CRON_LINE="0 22-23,0-9 * * *  bash $REPO_ROOT/scripts/cron.sh"
  # Remove existing entry first
  crontab -l 2>/dev/null | grep -v "duckbot-rag-memory/scripts/cron.sh" | crontab - || true
  (crontab -l 2>/dev/null; echo "$CRON_LINE") | crontab -
  echo "Cron installed:"
  crontab -l | grep "duckbot-rag-memory"
fi

echo ""
echo "=== Install complete ==="
echo ""
echo "Next steps:"
echo "  1. Edit .env to set OPENAI_API_KEY"
echo "  2. Run: .venv/bin/python -m src.cli doctor  (verify all green)"
echo "  3. Run: .venv/bin/python -m src.cli ingest ~/.openclaw/workspace/memory"
echo "  4. Run: .venv/bin/python -m src.cli query 'What did we decide about cloud-only models?'"
echo "  5. Push to GitHub: git remote add origin <url> && git push -u origin main"
