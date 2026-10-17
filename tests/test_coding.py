"""Phase E: coding diff-chains from git history."""
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.memory import Memory
from src import coding, llm_client
from tests.test_reconcile import KeywordEmbedder


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t",
                    "-c", "user.name=t", *args],
                   check=True, capture_output=True)


@pytest.fixture
def repo():
    r = Path(tempfile.mkdtemp(prefix="duckbot-git-"))
    _git(r, "init", "-q")
    (r / "a.py").write_text("x = 1\n")
    _git(r, "add", "-A"); _git(r, "commit", "-qm", "add a module")
    (r / "b.py").write_text("y = 2\n")
    _git(r, "add", "-A"); _git(r, "commit", "-qm", "add b module")
    (r / "a.py").write_text("x = 42\n")
    _git(r, "add", "-A"); _git(r, "commit", "-qm", "fix a value")
    return r


@pytest.fixture
def mem():
    tmp = Path(tempfile.mkdtemp(prefix="duckbot-coding-test-"))
    return Memory(persist_dir=tmp / "chroma", embedder=KeywordEmbedder())


async def test_chain_links_and_idempotency(repo, mem, monkeypatch):
    def _stub(messages, **k):
        user = messages[-1]["content"]
        subject = user.splitlines()[0].replace("COMMIT: ", "")
        return f"Decision record for: {subject}"
    monkeypatch.setattr(llm_client, "chat_completion", _stub)
    res = await coding.store_chain(mem, repo, model="hy-memory-30b-a3b-mlx")
    assert res["stored"] == 3 and res["fallback"] == 0
    res2 = await coding.store_chain(mem, repo, model="hy-memory-30b-a3b-mlx")
    assert res2["stored"] == 0 and res2["skipped"] == 3
    # chain: third commit (a.py again) must link back to the first (a.py)
    hashes = [c["hash"] for c in coding.iter_commits(repo)]
    recs = await coding.recall_coding(mem, "module changed", repo=repo.name)
    by_commit = {r["commit"]: r for r in recs}
    assert by_commit[hashes[2]]["chain_prev"] == hashes[0]
    assert by_commit[hashes[1]].get("chain_prev") is None


async def test_chain_fail_open_without_model(repo, mem):
    res = await coding.store_chain(mem, repo, model=None)
    assert res["stored"] == 3 and res["fallback"] == 3  # raw subjects kept


def test_iter_commits_missing_repo():
    assert coding.iter_commits("/nonexistent/path/xyz") == []
