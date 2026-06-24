"""
test_secret_scan.py — verify the pre-commit secret-scan guards real leaks.

duckbot-secret-scan: allowlist-file
(NOTE: This test file contains FAKE secret-shaped strings used as test
inputs. The pre-commit hook allows it via the marker above.)

Pattern: spin up throwaway git repos in /tmp, stage files with and
without secrets, and assert the scanner behaves correctly.

We don't import scripts/secret-scan.sh as Python (it's bash); we shell out.
That's fine — bash tests via subprocess are a well-understood pattern.

NOTE: This test file is itself a test fixture containing FAKE secret-shaped
strings (OpenAI/Anthropic/GitHub/MiniMax/AWS patterns used as test inputs).
The pre-commit hook must allowlist it. We use the `allowlist-file` marker
at the top — see scripts/secret-scan.sh for the rules.
"""

# duckbot-secret-scan: allowlist-file
from __future__ import annotations

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

SCAN_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "secret-scan.sh"


def _run_in_tmp_repo(tmp_path: Path, staged_files: dict[str, str]) -> subprocess.CompletedProcess:
    """Create a fresh git repo in tmp_path with the given staged files.

    `staged_files` maps relative path -> content.
    Returns the subprocess result of running the scan script.
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.check_call(["git", "init", "-q"], cwd=repo)
    subprocess.check_call(["git", "config", "user.email", "test@x"], cwd=repo)
    subprocess.check_call(["git", "config", "user.name", "test"], cwd=repo)

    # Symlink the scan script into the repo so the repo-root lookup works.
    (repo / "scripts").symlink_to(
        Path(__file__).resolve().parent.parent / "scripts"
    )

    for rel_path, content in staged_files.items():
        full = repo / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
        subprocess.check_call(["git", "add", rel_path], cwd=repo)

    return subprocess.run(
        ["bash", str(SCAN_SCRIPT)],
        cwd=repo,
        capture_output=True,
        text=True,
        env={**os.environ, "DUCKBOT_SKIP_SECRET_SCAN": ""},
    )


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    return tmp_path


# -----------------------------------------------------------------------------
# Positive cases — secrets that MUST be blocked
# -----------------------------------------------------------------------------


def test_blocks_openai_api_key(tmp_repo):
    r = _run_in_tmp_repo(
        tmp_repo,
        {"app.py": 'OPENAI_API_KEY = "sk-abcdefghijklmnopqrstuvwxyz1234567890XYZ"\n'},
    )
    assert r.returncode == 1
    assert "OpenAI API key" in r.stderr


def test_blocks_anthropic_api_key(tmp_repo):
    r = _run_in_tmp_repo(
        tmp_repo,
        {"app.py": 'key = "sk-ant-abcdefghijklmnopqrstuvwxyz1234567890XYZ1234"\n'},
    )
    assert r.returncode == 1
    assert "Anthropic API key" in r.stderr


def test_blocks_github_pat(tmp_repo):
    r = _run_in_tmp_repo(
        tmp_repo,
        {"app.py": 'GITHUB_TOKEN = "ghp_abcdefghijklmnopqrstuvwxyz1234567890"\n'},
    )
    assert r.returncode == 1
    assert "GitHub PAT" in r.stderr


def test_blocks_minimax_api_key(tmp_repo):
    r = _run_in_tmp_repo(
        tmp_repo,
        {"app.py": 'MINIMAX_API_KEY = "MiniMax-abcdefghijklmnopqrstuvwxyz1234"\n'},
    )
    assert r.returncode == 1
    assert "MiniMax API key" in r.stderr


def test_blocks_aws_access_key(tmp_repo):
    r = _run_in_tmp_repo(
        tmp_repo,
        {"app.py": 'AWS_ACCESS_KEY_ID = "AKIAIOSFODNN7EXAMPLE"\n'},
    )
    assert r.returncode == 1
    assert "AWS access key" in r.stderr


def test_blocks_bearer_token(tmp_repo):
    r = _run_in_tmp_repo(
        tmp_repo,
        {"app.py": 'headers = {"Authorization": "Bearer abcdefghijklmnopqrstuvwxyz1234"}\n'},
    )
    assert r.returncode == 1
    assert "Bearer token" in r.stderr


def test_blocks_private_rsa_key(tmp_repo):
    r = _run_in_tmp_repo(
        tmp_repo,
        {"key.pem": "-----BEGIN RSA PRIVATE KEY-----\nMIIEogIBAAKCAQEA...\n-----END RSA PRIVATE KEY-----\n"},
    )
    assert r.returncode == 1
    assert "Private RSA key" in r.stderr


def test_blocks_private_openssh_key(tmp_repo):
    r = _run_in_tmp_repo(
        tmp_repo,
        {"key": "-----BEGIN OPENSSH PRIVATE KEY-----\nb3BlbnNzaC1rZXktdjEAAAAA...\n"},
    )
    assert r.returncode == 1
    assert "Private OpenSSH key" in r.stderr


# -----------------------------------------------------------------------------
# Path blocks
# -----------------------------------------------------------------------------


def test_blocks_dotenv_file(tmp_repo):
    r = _run_in_tmp_repo(tmp_repo, {".env": "FOO=bar\n"})
    assert r.returncode == 1
    assert ".env" in r.stderr or "Forbidden path" in r.stderr


def test_blocks_data_chroma_dir(tmp_repo):
    r = _run_in_tmp_repo(tmp_repo, {"data/chroma/index.bin": "x"})
    assert r.returncode == 1
    assert "Forbidden path" in r.stderr


def test_blocks_venv_dir(tmp_repo):
    r = _run_in_tmp_repo(tmp_repo, {".venv/lib/foo.py": "x = 1\n"})
    assert r.returncode == 1
    assert "Forbidden path" in r.stderr


# -----------------------------------------------------------------------------
# Negative cases — clean files MUST pass
# -----------------------------------------------------------------------------


def test_passes_clean_python_file(tmp_repo):
    r = _run_in_tmp_repo(
        tmp_repo,
        {"app.py": "def hello():\n    return 'world'\n"},
    )
    assert r.returncode == 0, f"clean file should pass, got:\n{r.stderr}"


def test_passes_clean_markdown(tmp_repo):
    r = _run_in_tmp_repo(
        tmp_repo,
        {"README.md": "# Hello\n\nThis is a clean readme.\n"},
    )
    assert r.returncode == 0


def test_passes_short_placeholder_keys(tmp_repo):
    """Short placeholders like 'sk-EXAMPLE' or test fixtures should pass."""
    r = _run_in_tmp_repo(
        tmp_repo,
        {"tests.py": 'API_KEY = "sk-TEST"\n'},
    )
    assert r.returncode == 0, f"placeholder should pass, got:\n{r.stderr}"


def test_passes_minimax_example_key(tmp_repo):
    """MiniMax's example/test keys from docs."""
    r = _run_in_tmp_repo(
        tmp_repo,
        {"docs.py": '# Example: MINIMAX_API_KEY = "sk-mm-..." # placeholder in docs\n'},
    )
    # The comment line is short enough to not match. Should pass.
    assert r.returncode == 0


def test_passes_license_block_with_keyword(tmp_repo):
    """The word 'key' or 'secret' in plain English shouldn't trigger."""
    r = _run_in_tmp_repo(
        tmp_repo,
        {"notes.md": "The secret to good code is naming things well. That's the key insight.\n"},
    )
    assert r.returncode == 0


# -----------------------------------------------------------------------------
# Opt-out
# -----------------------------------------------------------------------------


def test_skip_env_var_allows_override(tmp_repo):
    """DUCKBOT_SKIP_SECRET_SCAN=1 should let the commit through (with warning)."""
    repo = tmp_repo / "repo"
    repo.mkdir()
    subprocess.check_call(["git", "init", "-q"], cwd=repo)
    subprocess.check_call(["git", "config", "user.email", "test@x"], cwd=repo)
    subprocess.check_call(["git", "config", "user.name", "test"], cwd=repo)
    (repo / "scripts").symlink_to(Path(__file__).resolve().parent.parent / "scripts")
    (repo / "leak.py").write_text('OPENAI_API_KEY = "sk-abcdefghijklmnopqrstuvwxyz1234567890XYZ"\n')
    subprocess.check_call(["git", "add", "leak.py"], cwd=repo)

    r = subprocess.run(
        ["bash", str(SCAN_SCRIPT)],
        cwd=repo,
        capture_output=True,
        text=True,
        env={**os.environ, "DUCKBOT_SKIP_SECRET_SCAN": "1"},
    )
    assert r.returncode == 0
    assert "WARNING" in r.stderr  # operator still sees the warning
