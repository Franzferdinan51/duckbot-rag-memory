"""
connectors/ — framework adapters for the DuckBot brain.

This package exposes the brain (Layers 0-5) to other agent frameworks:

  - openclaw — Model Context Protocol (MCP) server config + tool definitions
               for OpenClaw's MCP integration. Stdlib HTTP transport.
  - hermes   — Python-import-friendly facade for Hermes agents (Telegram-polling
               gateway, not HTTP). Includes a CLI shim so Hermes can shell out
               instead of importing.

Why a dedicated package? The brain's core modules (memory, graph, blocks,
entities, injection_scan) should stay framework-agnostic. Connectors adapt
those modules to specific transport protocols and tool conventions.

No LLM, no paid services, no surprises. Local Python + SQLite.
"""
