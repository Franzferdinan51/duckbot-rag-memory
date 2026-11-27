"""
connectors/ — framework adapters for the DuckBot brain.

This package exposes the brain (Layers 0-5) to other agent frameworks:

  - hermes   — Python-import-friendly facade for Hermes agents (Telegram-polling
               gateway, not HTTP). Includes a CLI shim so Hermes can shell out
               instead of importing.

(OpenClaw adapters were removed in v0.19.0 — the brain serves Hermes + Juno only.)

Why a dedicated package? The brain's core modules (memory, graph, blocks,
entities, injection_scan) should stay framework-agnostic. Connectors adapt
those modules to specific transport protocols and tool conventions.

No LLM, no paid services, no surprises. Local Python + SQLite.
"""
