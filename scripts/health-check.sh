#!/bin/bash
# DuckBot RAG Memory Health Check Script

echo "=== DuckBot RAG Memory Health Check ==="
echo "Date: $(date)"
echo ""

cd /Users/duckets/duckbot-rag-memory

echo "1. Testing MCP server stdio connection..."

# Test Python imports and server can be started
python3 << 'PYEOF'
import sys
import asyncio

async def test_mcp():
    from mcp import Client, StdioServerParameters
    from mcp.transports.stdio import StdioServerTransport
    
    server = StdioServerParameters(
        cmd="python",
        args=["-m", "duckbot_rag_memory.server"],
        cwd="/Users/duckets/duckbot-rag-memory"
    )
    transport = StdioServerTransport(server)
    
    try:
        async with Client(transport, None) as client:
            await client.initialize()
            print("✓ MCP client initialized successfully")
            
            # List tools
            tools = await client.list_tools()
            print(f"✓ MCP has {len(tools)} tools: {[t.name for t in tools]}")
            
            # Test brain_stats
            print("\n2. Running brain_stats...")
            try:
                result = await client.call_tool("brain_stats")
                if hasattr(result, 'content'):
                    print(f"✓ brain_stats succeeded, content type: {type(result.content)}")
                    if isinstance(result.content, str):
                        print(f"  Output: {result.content[:200] if result.content else 'empty'}...")
                    elif hasattr(result.content, '__len__'):
                        print(f"  Output length: {len(result.content)} items")
            except Exception as e:
                print(f"✗ brain_stats failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Test brain_block_list
            print("\n3. Running brain_block_list...")
            try:
                result = await client.call_tool("brain_block_list")
                print(f"✓ brain_block_list returned: {type(result.content)}")
            except Exception as e:
                print(f"✗ brain_block_list failed: {e}")
            
            print("\n=== Health Check Complete ===")
            return 0
            
    except Exception as e:
        print(f"✗ MCP connection failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

exit_code = asyncio.run(test_mcp())
sys.exit(exit_code)
PYEOF

echo ""
echo "=== Script Exit Code: $? ==="
