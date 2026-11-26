"""v0.17.0 reconciliation guards (2026-11-26).

These tests exist because each of these drifted once and caused real
damage; they are cheap tripwires against the same drift returning:

1. The Hermes plugin lived in divergent copies (deployed != repo
   `hermes/plugins/` != stale profile copies) — the two repo copies
   must stay byte-identical.
2. The consolidation writer was pinned to Hy-Memory-30B, whose nightly
   load failed for 18 straight nights while reporting success. The
   writer pin must stay on the light model.
3. The update-ops prompt was too loose for a small generalist writer
   (it returned op names instead of op objects) — it must keep its
   concrete example.
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def test_plugin_copies_byte_identical():
    a = (REPO / "hermes/plugins/duckbot_brain/__init__.py").read_bytes()
    b = (REPO / "src/plugins/memory/duckbot_brain/__init__.py").read_bytes()
    assert a == b, "plugin copies diverged — hermes/plugins/ is canonical"
    ya = (REPO / "hermes/plugins/duckbot_brain/plugin.yaml").read_bytes()
    yb = (REPO / "src/plugins/memory/duckbot_brain/plugin.yaml").read_bytes()
    assert ya == yb, "plugin.yaml copies diverged"


def test_writer_pin_is_light_model():
    for rel in ["scripts/nightly_hy_pass.py", "scripts/consolidate_job.py",
                "scripts/ingest_sessions.py", "scripts/mine_experience.py",
                "scripts/coding_chain.py"]:
        text = (REPO / rel).read_text()
        assert "lfm3-omni-2b" in text, f"{rel} lost the light writer pin"
        assert 'MODEL = "hy-memory-30b' not in text
        assert 'MODEL="hy-memory-30b' not in text


def test_update_prompt_carries_example():
    from src.hy_memory import HY_MEMORY_UPDATE_PROMPT
    assert '"op": "add"' in HY_MEMORY_UPDATE_PROMPT
    assert "never a bare string" in HY_MEMORY_UPDATE_PROMPT


def test_gate_model_is_d1_2b():
    from src.gate import DECISION_MODEL
    assert DECISION_MODEL == "d1-2b"
