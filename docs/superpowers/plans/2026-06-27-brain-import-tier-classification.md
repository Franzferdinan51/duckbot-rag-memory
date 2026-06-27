# Brain Import Tier Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make generic `brain_import` classify decisions, preferences, and setup notes as semantic memories instead of procedural memories.

**Architecture:** Keep exported-brain round trips unchanged because exported chunks carry explicit tier metadata. Adjust only the generic markdown-section fallback classifier used when importing arbitrary `## Heading` documents.

**Tech Stack:** Python async MCP handler tests, Chroma-isolated CLI E2E smoke, existing `Memory.remember(force_tier=...)` path.

---

### Task 1: Add Regression Coverage

**Files:**
- Modify: `tests/test_bugfixes_v0_11_3.py`

- [x] **Step 1: Write a generic import tier test**

Add this test near the existing brain export/import tests:

```python
def test_brain_import_generic_sections_classifies_durable_facts_as_semantic(tmp_path, monkeypatch):
    """Generic markdown imports should put decisions/preferences/setup in semantic.

    Export round trips keep explicit tier metadata, but arbitrary user markdown
    relies on the fallback classifier. The public schema advertises durable facts
    as semantic and only rules/how-tos/imperatives as procedural.
    """
    import asyncio
    from types import SimpleNamespace

    from src.mcp_server import handle_brain_import

    import_path = tmp_path / "import.md"
    import_path.write_text(
        "\n".join([
            "# Import",
            "",
            "## Decision: local-first embeddings",
            "",
            "Use LM Studio by default.",
            "",
            "## Preference: dark mode",
            "",
            "Duckets prefers dark mode.",
            "",
            "## Setup: watcher",
            "",
            "Installed watcher daemon.",
            "",
            "## How to restart watcher",
            "",
            "Always use scripts/start-watcher.sh.",
        ]),
        encoding="utf-8",
    )

    calls = []

    class _FakeImportMemory:
        async def remember(self, text, source_path, metadata=None, force_tier=None):
            calls.append({"text": text, "force_tier": force_tier})
            return SimpleNamespace(stored=True)

    monkeypatch.setattr("src.memory.Memory", _FakeImportMemory)
    monkeypatch.setattr("src.memory._DEFAULT_MEMORY", None)

    result = asyncio.run(handle_brain_import({"in_path": str(import_path)}))

    assert result["stored"] == 4
    assert [call["force_tier"] for call in calls] == [
        "semantic",
        "semantic",
        "semantic",
        "procedural",
    ]
```

- [x] **Step 2: Run the failing test**

Run:

```bash
./.venv/bin/python -m pytest -q tests/test_bugfixes_v0_11_3.py::test_brain_import_generic_sections_classifies_durable_facts_as_semantic
```

Expected before implementation: FAIL because decision/preference/setup sections are currently forced to procedural.

### Task 2: Fix Generic Import Classifier

**Files:**
- Modify: `src/mcp_server.py`

- [x] **Step 1: Update the import schema description**

Change the `brain_import` tool description so it says:

```text
semantic if 'decision/preference/setup', episodic if 'YYYY-MM-DD',
procedural if 'rule/how to/always/never/must/should not/do not',
else semantic
```

- [x] **Step 2: Update the handler docstring**

Change the tier bullets in `handle_brain_import()` to match the schema:

```text
- 'YYYY-MM-DD' or 'today' / 'yesterday' -> episodic
- 'rule' / 'how to' / 'always' / 'never' / 'must' / 'should not' / 'do not' -> procedural
- 'decision' / 'preference' / 'setup' -> semantic
- otherwise -> semantic
```

- [x] **Step 3: Update `_classify()` keys**

Replace the classifier key lists with:

```python
PROC_KEYS = ("rule", "how to", "always", "never", "must", "should not", "do not")
SEM_KEYS = ("preference", "prefers", "likes", "decision", "decided", "setup", "installed")
EPIS_KEYS = ("today", "yesterday", "log", "session")
```

Then check `SEM_KEYS` before falling back to semantic:

```python
if any(k in head for k in PROC_KEYS):
    return "procedural"
if any(k in head for k in SEM_KEYS):
    return "semantic"
return "semantic"
```

### Task 3: Verify End To End And Push

**Files:**
- Modify: `src/mcp_server.py`
- Modify: `tests/test_bugfixes_v0_11_3.py`
- Add: `docs/superpowers/plans/2026-06-27-brain-import-tier-classification.md`

- [x] **Step 1: Run focused tests**

Run:

```bash
./.venv/bin/python -m pytest -q tests/test_bugfixes_v0_11_3.py -k "brain_import"
```

Expected: all selected tests pass.

- [x] **Step 2: Run isolated CLI E2E import/query smoke**

Run:

```bash
tmp=$(mktemp -d)
cat > "$tmp/import.md" <<'EOF'
# Tiny Import

## Decision: import smoke

We decided DuckBot memory import smoke works.
EOF
DUCKBOT_CHROMA_DIR="$tmp/chroma" DUCKBOT_EMBEDDING=local \
  ./.venv/bin/python -m src.cli import "$tmp/import.md"
DUCKBOT_CHROMA_DIR="$tmp/chroma" DUCKBOT_EMBEDDING=local \
  ./.venv/bin/python -m src.cli query "DuckBot memory import smoke"
```

Expected: import succeeds and query result shows `tier=semantic`.

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
git add src/mcp_server.py tests/test_bugfixes_v0_11_3.py docs/superpowers/plans/2026-06-27-brain-import-tier-classification.md
git commit -m "Fix brain import tier inference"
git push origin main
```
