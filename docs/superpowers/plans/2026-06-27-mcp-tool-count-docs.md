# MCP Tool Count Docs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep living onboarding docs aligned with the actual MCP stdio server tool count so users do not think a healthy install is missing tools.

**Architecture:** Use the live `src.mcp_server.TOOLS` list and an end-to-end MCP `tools/list` request as evidence. Update only current docs/source comments that describe present behavior; leave historical changelog entries alone.

**Tech Stack:** Python pytest, subprocess-based MCP stdio JSON-RPC smoke test, Markdown docs.

---

### Task 1: Add Drift Regression

**Files:**
- Create: `tests/test_mcp_docs_truth.py`

- [x] **Step 1: Write the failing docs-count test**

```python
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
```

- [x] **Step 2: Run the test before docs edits**

Run:

```bash
./.venv/bin/python -m pytest -q tests/test_mcp_docs_truth.py::test_living_docs_match_mcp_tool_count
```

Expected before docs edits: FAIL on the stale 66/64 claims.

### Task 2: Update Living Docs

**Files:**
- Modify: `README.md`
- Modify: `INSTALL.md`
- Modify: `AGENTS.md`
- Modify: `docs/INTEGRATION.md`
- Modify: `docs/PLUGIN_SURFACE.md`
- Modify: `src/extensions/tools.py`

- [x] **Step 1: Replace present-tense 66-tool claims**

Change current onboarding claims from `66 tools` to `67 tools` in README, INSTALL, AGENTS, and integration docs.

- [x] **Step 2: Replace stale 64-tool plugin-surface claims**

In `docs/PLUGIN_SURFACE.md`, make the canonical MCP row and heading say `67 tools`, and update the note that referenced the old canonical 64-tool surface.

- [x] **Step 3: Update source docstring**

In `src/extensions/tools.py`, update the top-level comment that says the canonical MCP server has 66 tools to 67 tools.

### Task 3: End-To-End Verification And Push

**Files:**
- Create: `tests/test_mcp_docs_truth.py`
- Modify: `README.md`
- Modify: `INSTALL.md`
- Modify: `AGENTS.md`
- Modify: `docs/INTEGRATION.md`
- Modify: `docs/PLUGIN_SURFACE.md`
- Modify: `src/extensions/tools.py`
- Add: `docs/superpowers/plans/2026-06-27-mcp-tool-count-docs.md`

- [x] **Step 1: Run docs-count test**

Run:

```bash
./.venv/bin/python -m pytest -q tests/test_mcp_docs_truth.py
```

Expected: PASS.

- [x] **Step 2: Run real MCP stdio end-to-end smoke**

Run:

```bash
printf '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}\n' \
  | ./.venv/bin/python -m src.mcp_server >/tmp/duckbot-mcp-list.out
./.venv/bin/python - <<'PY'
import json
from pathlib import Path
data = json.loads(Path("/tmp/duckbot-mcp-list.out").read_text())
tools = data["result"]["tools"]
assert len(tools) == 67, len(tools)
print(len(tools))
PY
```

Expected: prints `67`.

- [x] **Step 3: Run full verification**

Run:

```bash
./.venv/bin/python -m pytest -q
./.venv/bin/python -m compileall -q src
git diff --check
bash scripts/secret-scan.sh
```

Expected: all pass.

- [ ] **Step 4: Commit and push**

Run:

```bash
git add README.md INSTALL.md AGENTS.md docs/INTEGRATION.md docs/PLUGIN_SURFACE.md src/extensions/tools.py tests/test_mcp_docs_truth.py docs/superpowers/plans/2026-06-27-mcp-tool-count-docs.md
git commit -m "Align MCP tool count docs with live server"
git push origin main
```
