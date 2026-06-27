from pathlib import Path

from src.mcp_server import TOOLS


ROOT = Path(__file__).resolve().parent.parent
LIVING_DOCS = [
    ROOT / "README.md",
    ROOT / "INSTALL.md",
    ROOT / "AGENTS.md",
    ROOT / "docs" / "INTEGRATION.md",
    ROOT / "docs" / "PLUGIN_SURFACE.md",
    ROOT / "src" / "extensions" / "tools.py",
]


def test_living_docs_match_mcp_tool_count():
    """Living docs should not drift from the canonical MCP tools/list surface."""
    expected = len(TOOLS)
    stale_patterns = [
        "66 tools",
        "66-tool",
        "**64**",
        "canonical 64-tool",
        "64-tool",
        "full 64",
    ]

    for path in LIVING_DOCS:
        text = path.read_text(encoding="utf-8")
        assert f"{expected} tools" in text or path.name == "tools.py"
        for pattern in stale_patterns:
            assert pattern not in text, f"{path.relative_to(ROOT)} still contains {pattern!r}"
