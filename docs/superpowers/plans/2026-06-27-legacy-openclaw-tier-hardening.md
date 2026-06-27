# Legacy OpenClaw Tier Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the legacy OpenClaw adapter treat whitespace-only tier input as unset, while still rejecting truly invalid tier names, so the older OpenClaw path matches the validated Brain and MCP behavior.

**Architecture:** Keep tier normalization centralized in `src/tier.py` and reuse that helper at the adapter boundary instead of hand-rolled string checks. Preserve the adapter's clean error shape for bad tier names, but let blank strings fall through as `None` so OpenClaw callers do not trip a ValueError on harmless whitespace.

**Tech Stack:** Python 3.12, pytest, legacy OpenClaw adapter, OpenClaw CLI shim.

---

### Task 1: Normalize legacy OpenClaw tier validation

**Files:**
- Modify: `src/connectors/openclaw.py:479-520, 707-810`
- Test: `tests/test_openclaw_shim.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_recall_ignores_whitespace_tier(monkeypatch):
    fake_brain = MagicMock()
    fake_brain.recall.return_value = []
    monkeypatch.setattr(openclaw, "Brain", lambda: fake_brain)

    out = openclaw.handle("brain_recall", {"query": "what changed", "tier": "   "})

    assert "error" not in out
    fake_brain.recall.assert_called_once()
    assert fake_brain.recall.call_args.kwargs["tier"] is None
```

```python
def test_recall_rejects_invalid_tier(monkeypatch):
    fake_brain = MagicMock()
    monkeypatch.setattr(openclaw, "Brain", lambda: fake_brain)

    out = openclaw.handle("brain_recall", {"query": "what changed", "tier": "not-a-tier"})

    assert out["error"].startswith("tier must be one of")
```

- [ ] **Step 2: Run the targeted tests and confirm the current adapter rejects whitespace as invalid**

Run:

```bash
./.venv/bin/python -m pytest tests/test_openclaw_shim.py -k "tier" -v
```

Expected: the new whitespace-tier test fails with the adapter's current `"tier must be one of ..."` error before the implementation change.

- [ ] **Step 3: Reuse the shared tier helper in the adapter**

```python
from src.tier import coerce_optional_tier


def _validate_tier(args: dict, tool_name: str) -> dict | None:
    try:
        tier = coerce_optional_tier(args.get("tier"))
    except ValueError:
        return {
            "error": f"tier must be one of {list(_VALID_TIERS)}, got {args.get('tier')!r}",
            "tool": tool_name,
        }
    args["tier"] = tier.value if tier is not None else None
    return None
```

```python
tier = args.get("tier")
tier = coerce_optional_tier(tier)
results = brain.recall(
    query=query,
    k=args.get("k", 5),
    tier=tier,
    min_importance=args.get("min_importance"),
    rerank=args.get("rerank"),
    decay=args.get("decay"),
    tier_priors=args.get("tier_priors"),
    tier_priors_overrides=tpo,
    fsrs=args.get("fsrs"),
)
```

- [ ] **Step 4: Re-run the targeted tests and verify they pass**

Run:

```bash
./.venv/bin/python -m pytest tests/test_openclaw_shim.py -k "tier" -v
```

Expected: whitespace tier is treated as omitted (`None`) and invalid tier names still return a clean error dict instead of a traceback.

### Task 2: Release gate

**Files:**
- Modify: `CHANGELOG.md` only if the adapter change is shipped in this pass

- [ ] **Step 1: Run the full verification set**

Run:

```bash
./.venv/bin/python -m pytest -q
bash scripts/secret-scan.sh
git diff --check
```

- [ ] **Step 2: Confirm the OpenClaw shim still behaves after the adapter change**

Run:

```bash
./.venv/bin/python -m pytest tests/test_openclaw_shim.py -q
```

- [ ] **Step 3: Commit and push the verified change**

```bash
git add src/connectors/openclaw.py tests/test_openclaw_shim.py docs/superpowers/plans/2026-06-27-legacy-openclaw-tier-hardening.md
git commit -m "Harden legacy OpenClaw tier validation"
git push origin main
```
