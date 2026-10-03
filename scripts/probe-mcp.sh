#!/bin/bash
# Probe MCP stdio server with brain_stats and brain_block_list
cd ~/duckbot-rag-memory
export PATH="$HOME/duckbot-rag-memory/.venv/bin:$PATH"
export PYTHONPATH="$HOME/duckbot-rag-memory/src:$PYTHONPATH"

python3 -c "
import sys, json
from mcp_server import MCPServer

server = MCPServer()
req1 = json.dumps({'jsonrpc':'2.0','id':1,'method':'tools/call','params':{'name':'brain_stats','arguments':{}}})
req2 = json.dumps({'jsonrpc':'2.0','id':2,'method':'tools/call','params':{'name':'brain_block_list','arguments':{}}})

# Each JSON-RPC request on its own line to stdin
for req in [req1, req2]:
    print(req, flush=True)
" 2>&1 | head -80
