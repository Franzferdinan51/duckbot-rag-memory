# Doctor JSON CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `python -m src.cli doctor --json` a supported scripted health-check command instead of an argparse error.

**Architecture:** Keep the existing doctor check builder as the single source of truth. Add a JSON branch to `cmd_doctor()` and register `--json` on the `doctor` subparser so CLI users and tests exercise the same path.

**Tech Stack:** Python argparse, existing `build_doctor_checks_async()`, pytest subprocess regression coverage.

---

### Task 1: Add Regression Coverage

**Files:**
- Modify: `tests/test_bugfixes_v0_11_3.py`

- [x] **Step 1: Write the failing subprocess test**

Add this test near the existing CLI doctor tests:

```python
def test_cli_doctor_json_flag_outputs_parseable_json(monkeypatch):
    """doctor --json is the scripted health-check path and must not argparse-fail."""
    import json
    import os
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["DUCKBOT_EMBEDDING"] = "local"
    env["DUCKBOT_DISABLE_IMPORT_SIDE_EFFECTS"] = "1"

    r = subprocess.run(
        [sys.executable, "-m", "src.cli", "doctor", "--json"],
        cwd=str(root),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert r.returncode in (0, 1), r.stderr
    data = json.loads(r.stdout)
    assert "ok" in data
    assert isinstance(data["checks"], list)
    assert all({"name", "value", "ok"} <= set(check) for check in data["checks"])
    assert "unrecognized arguments" not in r.stderr
```

- [x] **Step 2: Run the failing test**

Run:

```bash
./.venv/bin/python -m pytest -q tests/test_bugfixes_v0_11_3.py::test_cli_doctor_json_flag_outputs_parseable_json
```

Expected before implementation: FAIL because argparse rejects `--json`.

### Task 2: Implement Doctor JSON Output

**Files:**
- Modify: `src/cli.py`

- [x] **Step 1: Add JSON output in `cmd_doctor()`**

Replace the body with:

```python
def cmd_doctor(args: argparse.Namespace) -> int:
    """Sanity check: env, deps, store."""
    checks, all_ok = asyncio.run(build_doctor_checks_async())
    if getattr(args, "json", False):
        print(json.dumps({
            "ok": all_ok,
            "checks": [
                {"name": name, "value": value, "ok": ok}
                for name, value, ok in checks
            ],
        }, indent=2, default=str))
        return 0 if all_ok else 1

    max_name = max(len(c[0]) for c in checks)
    for name, value, ok in checks:
        marker = "✓" if ok else "✗"
        print(f"  {marker} {name.ljust(max_name)}  {value}")
    return 0 if all_ok else 1
```

- [x] **Step 2: Register the argparse flag**

Change the doctor parser setup to:

```python
p_doc = sub.add_parser("doctor", help="check env + deps")
p_doc.add_argument("--json", action="store_true", help="output as JSON")
p_doc.set_defaults(func=cmd_doctor)
```

### Task 3: Verify And Commit

**Files:**
- Modify: `src/cli.py`
- Modify: `tests/test_bugfixes_v0_11_3.py`
- Add: `docs/superpowers/plans/2026-06-27-doctor-json-cli.md`

- [x] **Step 1: Run focused tests**

Run:

```bash
./.venv/bin/python -m pytest -q tests/test_bugfixes_v0_11_3.py -k "doctor_json or doctor"
```

Expected: all selected tests pass.

- [x] **Step 2: Run command smoke tests**

Run:

```bash
DUCKBOT_EMBEDDING=local ./.venv/bin/python -m src.cli doctor --json
```

Expected: JSON object with `ok` and `checks`.

- [x] **Step 3: Run full verification**

Run:

```bash
./.venv/bin/python -m pytest -q
git diff --check
bash scripts/secret-scan.sh
```

Expected: test suite passes, diff check is clean, secret scan is clean.

- [ ] **Step 4: Commit and push**

Run:

```bash
git add src/cli.py tests/test_bugfixes_v0_11_3.py docs/superpowers/plans/2026-06-27-doctor-json-cli.md
git commit -m "Support JSON output for doctor command"
git push origin main
```
