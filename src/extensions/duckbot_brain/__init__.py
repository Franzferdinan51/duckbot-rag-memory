"""
extensions/duckbot_brain/__init__.py — re-export the OpenClaw adapter and
Hermes provider so either loader can `from src.extensions.duckbot_brain
import ...`.

MIT License — see LICENSE in the repository root.
"""
from .adapter import (
    handle_request,
    _tool_schemas,
    _call_tool,
)
