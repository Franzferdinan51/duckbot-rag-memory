#!/usr/bin/env bash
# brain-recall.sh — query the DuckBot RAG memory from any shell.
#
# Convenience wrapper around duckbot-ask. Same arguments, same output
# formats. Lives in the brain's own scripts/ so it's versioned with the
# brain, and exists for discoverability (`ls scripts/brain-*`).
#
# Most users should call scripts/duckbot-ask directly — this is just
# an alias for those who prefer the "brain-recall" verb.
#
# Usage:
#   brain-recall "your question"
#   brain-recall -n 5 -f compact "Duckets correction style"
#   brain-recall -f snippet "BATMAN container restart recipe"
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/duckbot-ask" "$@"
