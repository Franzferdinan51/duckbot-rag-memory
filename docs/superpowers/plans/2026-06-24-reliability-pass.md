# Reliability Pass Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair confirmed runtime and test reliability issues in the dashboard, watcher documentation, and shell query wrappers without changing the memory architecture.

**Architecture:** Preserve the dashboard's true 24-hour cutoff, but make its clock injectable at the report boundary so tests remain valid over time. Make script distribution match its documented direct-execution interface, and align help text with actual defaults.

**Tech Stack:** Python 3.12, pytest, Bash, ChromaDB.

---

### Task 1: Make dashboard time-window tests deterministic

**Files:**
- Modify: `src/dashboard.py`
- Modify: `tests/test_dashboard.py`

- [x] Add a failing test that passes a fixed `now` timestamp to `build_report()` and asserts fresh log events count while an event just beyond the cutoff does not.
- [x] Run `./.venv/bin/python -m pytest tests/test_dashboard.py -q` and verify the new API is unavailable before implementation.
- [x] Add an optional `now: float | None` parameter to `build_report()` and `_summarize_last_24h()`; use `time.time()` only when it is `None`.
- [x] Update existing fixture timestamps to be relative to a fixed test clock.
- [x] Re-run `./.venv/bin/python -m pytest tests/test_dashboard.py -q` and verify all dashboard tests pass.

### Task 2: Honour dashboard custom Chroma locations

**Files:**
- Modify: `src/dashboard.py`
- Modify: `tests/test_dashboard.py`

- [x] Add a failing test that supplies `chroma_path` and asserts the dashboard obtains store statistics from that exact directory.
- [x] Run that test and verify it fails because `chroma_path` is ignored.
- [x] Pass the explicit directory through `_get_chroma_stats_directly()` to `MemoryStore(persist_dir=...)` while preserving the default path when none is supplied.
- [x] Re-run the targeted test and verify it passes.

### Task 3: Repair the documented shell command interface

**Files:**
- Modify: `scripts/duckbot-ask` (mode and help text)
- Modify: `scripts/brain-recall.sh` (mode)
- Modify: `tests/test_duckbot_ask.py`
- Modify: `src/watcher.py`

- [x] Add a POSIX-only test that invokes each wrapper's `--help` directly and asserts it exits successfully; assert duckbot-ask help states the actual default `500` characters.
- [x] Run the targeted wrapper tests and verify direct execution fails while the scripts lack execute bits.
- [x] Mark both wrappers executable in Git and correct the help default from `200` to `500`.
- [x] Correct stale watcher docstrings that say polling defaults to two seconds; the implementation default is 300 seconds.
- [x] Re-run the targeted wrapper tests and verify they pass.

### Task 4: Validate and publish

**Files:**
- Modify: `CHANGELOG.md`

- [x] Add a concise unreleased entry describing the reliability fixes.
- [x] Run `./.venv/bin/python -m pytest -q`, `./.venv/bin/python -m compileall -q src`, `bash scripts/secret-scan.sh`, `./.venv/bin/python -m src.cli dashboard --json`, and each direct wrapper `--help` command.
- [ ] Inspect `git diff --check` and `git status --short`; stage only reliability-pass files.
- [ ] Commit with `fix: harden dashboard and shell wrappers` and push the verified commit to `origin/main`.
