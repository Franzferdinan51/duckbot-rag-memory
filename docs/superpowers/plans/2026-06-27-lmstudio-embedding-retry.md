# LM Studio Embedding Retry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make user-facing query flows tolerate transient LM Studio embedding transport failures instead of crashing the CLI on a one-off local server hiccup.

**Architecture:** Add a narrow retry loop inside `LMStudioEmbeddings.embed()` for transport errors and transient HTTP statuses. Keep non-transient model/auth errors visible so bad configuration still fails loudly.

**Tech Stack:** Python `httpx`, async pytest, existing CLI query E2E tests.

---

### Task 1: Pin The Runtime Failure

**Files:**
- Modify: `tests/test_v0_11_2_hotfix.py`

- [x] **Step 1: Add a transient transport regression**

Add `test_embed_retries_transient_transport_error()` to simulate `httpx.ReadError` on the first LM Studio `/embeddings` request and success on the second request.

- [x] **Step 2: Run the regression**

Run:

```bash
./.venv/bin/python -m pytest -q tests/test_v0_11_2_hotfix.py::TestEmbedEndToEndCaching::test_embed_retries_transient_transport_error
```

Expected after implementation: PASS with exactly two fake HTTP calls.

### Task 2: Add Narrow LM Studio Retry

**Files:**
- Modify: `src/embeddings.py`

- [x] **Step 1: Add retry settings**

Add `max_retries: int = 2` to `LMStudioEmbeddings`.

- [x] **Step 2: Retry only transient failures**

Retry `httpx.TransportError` and HTTP `429/500/502/503/504`, with short exponential backoff. Do not swallow final failures.

### Task 3: Verify End To End

**Files:**
- Modify: `src/embeddings.py`
- Modify: `tests/test_v0_11_2_hotfix.py`
- Add: `docs/superpowers/plans/2026-06-27-lmstudio-embedding-retry.md`

- [x] **Step 1: Run focused retry test**

Run:

```bash
./.venv/bin/python -m pytest -q tests/test_v0_11_2_hotfix.py::TestEmbedEndToEndCaching::test_embed_retries_transient_transport_error
```

- [x] **Step 2: Run real query E2E slice**

Run:

```bash
./.venv/bin/python -m pytest -q \
  tests/test_duckbot_ask.py::TestBrainIntegration::test_query_returns_structured_output \
  tests/test_duckbot_ask.py::TestBrainIntegration::test_query_returns_numbered_blocks \
  tests/test_duckbot_ask.py::TestBrainIntegration::test_highly_relevant_query_returns_useful_chunks \
  tests/test_duckbot_ask.py::TestBrainIntegration::test_compact_formatter_works_on_real_query
```

- [x] **Step 3: Run full verification**

Run:

```bash
./.venv/bin/python -m pytest -q
node --test extensions/duckbot-memory/test/shim.test.js
./.venv/bin/python -m compileall -q src
git diff --check
bash scripts/secret-scan.sh
```

- [ ] **Step 4: Commit and push**

Run:

```bash
git add src/embeddings.py tests/test_v0_11_2_hotfix.py docs/superpowers/plans/2026-06-27-lmstudio-embedding-retry.md
git commit -m "Retry transient LM Studio embedding failures"
git push origin main
```
