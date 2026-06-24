#!/usr/bin/env bash
# secret-scan.sh — pre-commit guard for the DuckBot brain repo.
#
# duckbot-secret-scan: allowlist-file
#
# Pattern source: MemPalace's `.pre-commit-config.yaml` (MIT).
# https://github.com/MemPalace/mempalace/blob/develop/.pre-commit-config.yaml
#
# Scans the *content of staged files* for things that should NEVER land
# in a commit:
#   - .env, .env.* files
#   - common API-key prefixes (OpenAI, Anthropic, MiniMax, GitHub, etc.)
#   - common secret env var names
#   - private key headers
#   - bearer tokens
#
# Exit codes:
#   0 — clean
#   1 — secrets detected (commit blocked)
#
# Install:
#   ln -sf ../../scripts/secret-scan.sh .git/hooks/pre-commit
# (Already done in this repo.)
#
# Skip with `git commit --no-verify` ONLY if you know what you're doing.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR/..")"
cd "$REPO_ROOT"

# Allow opt-out for emergency commits. Logged to stderr so it's visible.
if [ "${DUCKBOT_SKIP_SECRET_SCAN:-0}" = "1" ]; then
    echo "WARNING: DUCKBOT_SKIP_SECRET_SCAN=1 — secret scan skipped" >&2
    exit 0
fi

# What we're about to add to the repo.
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM 2>/dev/null || true)
if [ -z "$STAGED_FILES" ]; then
    exit 0
fi

# Patterns that always indicate a secret. Each line: <label>|<regex>
SECRET_PATTERNS=(
    "OpenAI API key|sk-[A-Za-z0-9]{20,}"
    "Anthropic API key|sk-ant-[A-Za-z0-9_-]{20,}"
    "GitHub PAT|ghp_[A-Za-z0-9]{36}"
    "GitHub fine-grained|github_pat_[A-Za-z0-9_]{60,}"
    "AWS access key|AKIA[0-9A-Z]{16}"
    "MiniMax API key|MiniMax-[A-Za-z0-9]{20,}"
    "Bearer token literal|Bearer[[:space:]]+[A-Za-z0-9._-]{20,}"
    "Generic high-entropy secret|(api[_-]?key|secret|token|password)[[:space:]]*[:=][[:space:]]*['\"][A-Za-z0-9._/+-]{16,}['\"]"
    "Private RSA key|-----BEGIN RSA PRIVATE KEY-----"
    "Private OpenSSH key|-----BEGIN OPENSSH PRIVATE KEY-----"
    "Private PGP key|-----BEGIN PGP PRIVATE KEY BLOCK-----"
)

FOUND=0

# Scan each staged file's working-tree content.
# We scan the staged file from the index via `git show :<path>` so we get
# exactly what would be committed (not whatever is in the working tree).
for f in $STAGED_FILES; do
    # Files can opt out of the scan with a top-of-file marker:
    #   # duckbot-secret-scan: allowlist-file
    #   """duckbot-secret-scan: allowlist-file"""
    #   // duckbot-secret-scan: allowlist-file
    # Use this for tests, fixtures, or files that intentionally contain
    # example API keys.
    if printf '%s\n' "$(git show ":$f" 2>/dev/null)" | head -30 | grep -qE "duckbot-secret-scan:[[:space:]]*allowlist-file"; then
        continue
    fi

    # Get the staged content for this file.
    STAGED_CONTENT=$(git show ":$f" 2>/dev/null || true)
    if [ -z "$STAGED_CONTENT" ]; then
        continue
    fi

    for pattern in "${SECRET_PATTERNS[@]}"; do
        label="${pattern%%|*}"
        regex="${pattern#*|}"
        # Search the staged content. Patterns starting with "-----" look like
        # grep flags, so we pass `--` to terminate flag parsing. We also use
        # grep -E for extended regex.
        HITS=$(printf '%s\n' "$STAGED_CONTENT" | grep -nE -e "$regex" -- 2>/dev/null || true)
        if [ -n "$HITS" ]; then
            echo "❌ Secret pattern detected: $label in $f" >&2
            echo "$HITS" | sed 's/^/    /' | head -5 >&2
            FOUND=1
        fi
    done
done

# Block known sensitive paths outright (even if their contents look clean).
FORBIDDEN_PATTERNS=(
    '^\.env$'
    '^\.env\.local$'
    '^\.env\.prod(uction)?$'
    '^\.env\.staging$'
    '^data/chroma/'
    '^data/watcher_state\.json$'
    '^data/ingest_history\.jsonl$'
    '^data/eval_history\.jsonl$'
    '^\.venv/'
    '^node_modules/'
)
for path_re in "${FORBIDDEN_PATTERNS[@]}"; do
    BAD=$(echo "$STAGED_FILES" | grep -E "$path_re" || true)
    if [ -n "$BAD" ]; then
        echo "❌ Forbidden path staged: $BAD" >&2
        FOUND=1
    fi
done

if [ $FOUND -ne 0 ]; then
    cat >&2 <<'EOF'

🛑 COMMIT BLOCKED. One or more secrets or forbidden paths were detected.

How to fix:
  1. If this is a real leak: ROTATE THE CREDENTIAL IMMEDIATELY before retrying.
  2. If it's a false positive (test fixture, example, etc.):
     - Use a placeholder like `sk-EXAMPLE-NOT-A-REAL-KEY` (short, won't match).
     - Add the file to .gitignore if it shouldn't be tracked at all.
  3. If you really must override (emergency hotfix only):
     DUCKBOT_SKIP_SECRET_SCAN=1 git commit --no-verify
     ⚠️ This is logged. Don't do it unless you're sure.

If you need help, ask the operator.
EOF
    exit 1
fi

exit 0
