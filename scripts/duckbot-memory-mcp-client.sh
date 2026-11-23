#!/usr/bin/env bash
# Hermes-side launcher for the duckbot brain (runs on the Hermes host).
#
# Hermes talks to the brain over ssh to the brain host (Robin). A plain ssh
# with no keepalive cannot tell a dead peer from a quiet one: when the far
# end vanishes (VPN/Tailscale flips, host sleep, network drop), the local
# ssh lingers for days and every brain call Hermes makes hangs into the
# dead pipe (the 2026-11-23 stall: two 1-2 day old zombie transports).
#
# ServerAliveInterval/CountMax make ssh probe the peer and exit when it is
# gone (~60s), so Hermes sees the transport close and respawns a fresh one
# on its next brain call. ConnectTimeout keeps a down host from hanging
# the spawn either.
set -euo pipefail

BRAIN_HOST="${DUCKBOT_BRAIN_HOST:-duckets@100.88.161.36}"
BRAIN_SCRIPT="${DUCKBOT_BRAIN_SCRIPT:-/home/duckets/duckbot-rag-memory/scripts/duckbot-memory-mcp.sh}"
SSH_KEY="${DUCKBOT_BRAIN_KEY:-$HOME/.ssh/id_ed25519}"

exec ssh \
    -o BatchMode=yes \
    -o IdentitiesOnly=yes \
    -i "$SSH_KEY" \
    -o StrictHostKeyChecking=accept-new \
    -o ConnectTimeout=10 \
    -o ServerAliveInterval=20 \
    -o ServerAliveCountMax=3 \
    -o TCPKeepAlive=yes \
    "$BRAIN_HOST" "$BRAIN_SCRIPT"
