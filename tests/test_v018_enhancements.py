"""v0.18.0 enhancement guards.

- The repo .env loader is shared (src/_env.py) and the brain worker uses
  it — before this, only the CLI loaded .env, so the worker ran without
  the DUCKBOT_* pins in production.
- scripts/backfill_graph.py watermark state round-trips, and a mini
  end-to-end run over a fake store feeds a real Graph.
- The canonical Robin wrappers encode the current operating rules:
  hourly sync targets Hermes only (owner's 2026-11-23 boundary), the
  nightly wrapper runs the graph backfill, a backup wrapper exists.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_env_loader_is_shared_and_idempotent():
    from src._env import load_repo_dotenv
    load_repo_dotenv()
    load_repo_dotenv()  # second call must be a no-op, not an error
    import src.cli as cli
    # cli keeps its historical entry point, now delegating to the shared one
    assert cli._load_dotenv is not None
    cli._load_dotenv()


def test_brain_worker_loads_repo_dotenv():
    src = (REPO / "src" / "brain_worker.py").read_text()
    assert "load_repo_dotenv" in src
    # and it happens before the Brain machinery is imported/used
    assert src.index("load_repo_dotenv()") < src.index("_get_brain")


def test_backfill_state_roundtrip(tmp_path, monkeypatch):
    mod = _load_script("backfill_graph")
    state = tmp_path / "state.json"
    monkeypatch.setattr(mod, "STATE_PATH", state)
    assert mod._load_state() == set()
    mod._save_state({"b", "a"})
    assert mod._load_state() == {"a", "b"}
    assert json.loads(state.read_text())["done_ids"] == ["a", "b"]


def test_backfill_feeds_graph_from_store(tmp_path, monkeypatch):
    mod = _load_script("backfill_graph")
    monkeypatch.setattr(mod, "STATE_PATH", tmp_path / "state.json")

    class FakeColl:
        def get(self, include=None, limit=1000, offset=0):
            if offset > 0:
                return {"ids": [], "documents": [], "metadatas": []}
            return {
                "ids": ["c1", "c2"],
                "documents": [
                    "Duckets runs Hermes on the Mac Studio every day.",
                    "x",  # too short -> skipped
                ],
                "metadatas": [{"source_path": "test"}, {"source_path": "test"}],
            }

    class FakeStore:
        def collection_for(self, tier):
            return FakeColl()

    monkeypatch.setattr(mod, "MemoryStore", lambda: FakeStore())
    from src.graph import Graph
    gpath = tmp_path / "graph.db"
    monkeypatch.setattr(mod, "graph_path", lambda: gpath)
    monkeypatch.setattr(sys, "argv", ["backfill_graph.py"])
    assert mod.main() == 0
    graph = Graph(str(gpath))
    names = {e.name for e in graph.list_entities(limit=1000)}
    assert "Duckets" in names  # the extractor's person pattern catches the owner
    # second run: watermark makes it a no-op scan
    assert mod.main() == 0


def test_wrappers_encode_current_rules():
    sync = (REPO / "scripts" / "brain-sync-wrapper.sh").read_text()
    assert "--target hermes" in sync
    assert "--target both" not in sync
    nightly = (REPO / "scripts" / "duckbot-nightly-brain.sh").read_text()
    assert "backfill_graph.py" in nightly
    backup = (REPO / "scripts" / "brain-backup-wrapper.sh").read_text()
    assert "backup-brain.sh" in backup
    assert "RC=$?" in backup  # exit code captured immediately, never masked
    bb = (REPO / "scripts" / "backup-brain.sh").read_text()
    assert "mktemp -t brain-backup" not in bb  # BSD-only syntax; GNU rejects it
