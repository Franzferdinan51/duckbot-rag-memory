"""Tests for the duckbot-ask brain-query helper.

Two layers of testing:

1. **Formatter unit tests** (`TestFormatScripts`) — directly test
   _format_snippet.py and _format_compact.py with synthetic input. These
   are pure-Python, no subprocess, no shell, no venv quirks.

2. **Integration tests** (`TestJsonFormat`, `TestCompactFormat`,
   `TestSnippetFormat`) — invoke `python -m src.cli query` directly
   via the repo's venv (no bash wrapper). The bash `duckbot-ask`
   wrapper is a thin shim around `python -m src.cli query` + these
   formatters, so testing those pieces covers the user's intent.

The bash wrapper itself is tested manually (it's a 100-line script that
just calls python with the right args + pipes). The complex logic
lives in the python formatters + the `src.cli query` command, both of
which are tested here.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "duckbot-ask"
RECALL_SCRIPT = REPO_ROOT / "scripts" / "brain-recall.sh"
SNIPPET_FORMATTER = REPO_ROOT / "scripts" / "_format_snippet.py"
COMPACT_FORMATTER = REPO_ROOT / "scripts" / "_format_compact.py"


def _python() -> str:
    """Path to the venv Python (Windows or POSIX layout)."""
    win = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
    pos = REPO_ROOT / ".venv" / "bin" / "python"
    return str(win if win.exists() else pos)


def _run_cli_query(query: str, n: int = 3, max_chars: int = 500) -> str:
    """Run `python -m src.cli query` directly. Returns stdout + stderr.

    The CLI prints the JSON header to stderr and the numbered blocks
    to stdout (so the blocks show up cleanly in a terminal when stderr
    is filtered out via `2>/dev/null`). We concatenate for tests.
    """
    result = subprocess.run(
        [_python(), "-m", "src.cli", "query", query, "-n", str(n), "--max-chars", str(max_chars)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(REPO_ROOT),
        env={**__import__("os").environ},
    )
    # Filter out chromadb noise from both stdout and stderr before
    # concatenating. Patterns include:
    #   - posthog telemetry errors: "Failed to send telemetry event"
    #   - chromadb index warnings: "Number of requested results N ..."
    #   - chromadb deletion noise: "Delete of nonexisting embedding ID"
    # These go to stdout/stderr and would break json.loads() / formatter output.
    def _clean(text: str) -> str:
        return "\n".join(
            line for line in text.splitlines()
            if "Failed to send telemetry event" not in line
            and "posthog" not in line.lower()
            and "Number of requested results" not in line
            and "updating n_results" not in line
            and "Delete of nonexisting embedding ID" not in line
            and "Delete of nonexisting" not in line
            and line.strip()
        )
    # Put stderr FIRST (the JSON header) then stdout (the blocks).
    return _clean(result.stderr) + "\n" + _clean(result.stdout)


def _run_formatter(script: Path, stdin_text: str, args: list[str] = None) -> str:
    """Run a python formatter script with stdin text and return stdout.

    `args` are passed as `python script.py <args...>` — they should be
    the COMPLETE arg list the script expects (e.g. snippet.py needs
    ["query", "500"]; compact.py needs ["query", "n", "500"]).
    """
    if args is None:
        # Default: just max_chars for snippet-style scripts.
        args = ["test-query", "500"]
    arg_list = [_python(), str(script)] + args
    result = subprocess.run(
        arg_list,
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return result.stdout


# ---------------------------------------------------------------------------
# Real brain query integration (uses LM Studio + Chroma)
# Skipped if the brain's venv doesn't have LM Studio or the API is offline.
# ---------------------------------------------------------------------------

LMSTUDIO_REACHABLE = None


def _lmstudio_reachable() -> bool:
    """Check if LM Studio is reachable + the brain can answer a query."""
    global LMSTUDIO_REACHABLE
    if LMSTUDIO_REACHABLE is not None:
        return LMSTUDIO_REACHABLE
    try:
        out = _run_cli_query("test", n=1, max_chars=50)
        LMSTUDIO_REACHABLE = '"fused_results"' in out and '"duration_seconds"' in out
    except Exception:
        LMSTUDIO_REACHABLE = False
    return LMSTUDIO_REACHABLE


# ---------------------------------------------------------------------------
# Formatter unit tests (no brain needed)
# ---------------------------------------------------------------------------

SAMPLE_OUTPUT = (
    '{"query": "test", "fused_results": 2}\n'
    '[1] (tier=episodic, rrf=0.0164, vec_rank=1)\n'
    'Source: /tmp/foo.md\n'
    'Section: ## Sample Header\n'
    '---\n'
    'This is the body of the first result.\n'
    '[2] (tier=episodic, rrf=0.0164, bm25_rank=1)\n'
    'Source: /tmp/bar.md\n'
    'Section: ## Another Header\n'
    '---\n'
    'This is the body of the second result.\n'
)


class TestFormatScripts:
    """Direct tests for _format_snippet.py and _format_compact.py."""

    def test_snippet_extracts_first_body(self):
        if not SNIPPET_FORMATTER.exists():
            pytest.skip("formatter not built yet")
        # snippet.py signature: <query> <max_chars>
        out = _run_formatter(SNIPPET_FORMATTER, SAMPLE_OUTPUT, args=["test-query", "500"])
        assert "This is the body of the first result" in out
        # Should NOT include the second result.
        assert "second result" not in out

    def test_snippet_handles_no_body(self):
        if not SNIPPET_FORMATTER.exists():
            pytest.skip("formatter not built yet")
        no_body = (
            '{"query": "x"}\n'
            '[1] (tier=episodic)\n'
            'Source: f\n'
            'Section: ## Just The Header\n'
            '---\n'  # nothing after this
        )
        out = _run_formatter(SNIPPET_FORMATTER, no_body, args=["test-query", "500"])
        # Falls back to section header with truncation note.
        assert "truncated" in out.lower() or "Just The Header" in out

    def test_compact_shapes_all_results(self):
        if not COMPACT_FORMATTER.exists():
            pytest.skip("formatter not built yet")
        # compact.py signature: <query> <n> <max_chars>
        out = _run_formatter(COMPACT_FORMATTER, SAMPLE_OUTPUT, args=["test-query", "2", "500"])
        assert "[1]" in out
        assert "[2]" in out
        assert "Sample Header" in out
        assert "Another Header" in out
        assert "results for:" in out  # footer

    def test_compact_truncates_long_paths(self):
        if not COMPACT_FORMATTER.exists():
            pytest.skip("formatter not built yet")
        long_path = "C:\\" + "very_long_subdir\\" * 10 + "file.md"
        sample = SAMPLE_OUTPUT.replace("/tmp/foo.md", long_path)
        out = _run_formatter(COMPACT_FORMATTER, sample, args=["test-query", "2", "500"])
        # Long paths get truncated with leading "..."
        assert "..." in out


# ---------------------------------------------------------------------------
# Bash wrapper structural tests (no execution, just sanity)
# ---------------------------------------------------------------------------

class TestBashWrapper:
    """Lightweight checks on the bash wrapper itself."""

    def test_wrapper_exists_and_executable(self):
        if not SCRIPT.exists():
            pytest.skip("duckbot-ask wrapper not built yet")
        # On Windows, file executable bits are unreliable (no x bit per se).
        # We just verify the file is present + non-empty + has a shebang or
        # bash shebang at the top so the calling shell knows how to run it.
        assert SCRIPT.stat().st_size > 100, f"{SCRIPT} too small"
        first_line = SCRIPT.read_text(encoding="utf-8").split("\n", 1)[0]
        assert first_line.startswith("#!") or first_line.startswith("@"), \
            f"missing shebang on {SCRIPT}: {first_line!r}"

    def test_wrapper_references_python_correctly(self):
        """The wrapper must NOT hardcode /c/... — it should detect paths."""
        if not SCRIPT.exists():
            pytest.skip("duckbot-ask wrapper not built yet")
        text = SCRIPT.read_text(encoding="utf-8")
        # Should reference the .env load + .venv/bin OR .venv/Scripts/python
        assert ".env" in text
        assert "Scripts" in text or "bin/python" in text
        # Should NOT have a hardcoded C:\ path that breaks cross-platform.
        assert "C:\\Users" not in text, \
            "wrapper has hardcoded Windows path — breaks cross-platform"

    def test_wrapper_handles_all_three_formats(self):
        if not SCRIPT.exists():
            pytest.skip("duckbot-ask wrapper not built yet")
        text = SCRIPT.read_text(encoding="utf-8")
        for fmt in ("json", "compact", "snippet"):
            assert fmt in text, f"wrapper missing format handler: {fmt}"

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX execute bits do not apply on Windows")
    @pytest.mark.parametrize("script", [SCRIPT, RECALL_SCRIPT])
    def test_wrapper_is_directly_executable(self, script):
        """Documented wrappers must run directly, not only via `bash script`."""
        result = subprocess.run(
            [str(script), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, result.stderr

    @pytest.mark.skipif(sys.platform == "win32", reason="POSIX shell help only")
    def test_wrapper_help_reports_actual_max_chars_default(self):
        result = subprocess.run(
            [str(SCRIPT), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=str(REPO_ROOT),
        )
        assert result.returncode == 0, result.stderr
        assert "default 500" in result.stdout


# ---------------------------------------------------------------------------
# Integration tests (require LM Studio reachable)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not _lmstudio_reachable(),
    reason="LM Studio not reachable; brain integration tests need a live embed endpoint",
)
class TestBrainIntegration:
    """End-to-end: query the brain via python -m src.cli query."""

    def test_query_returns_structured_output(self):
        out = _run_cli_query("what did we decide about cloud-only models")
        first_line = out.split("\n")[0]
        # First line should be valid JSON
        data = json.loads(first_line)
        assert "query" in data
        assert "fused_results" in data

    def test_query_returns_numbered_blocks(self):
        out = _run_cli_query("what did we decide about cloud-only models", n=3)
        assert "[1]" in out
        assert "[2]" in out
        assert "[3]" in out

    def test_highly_relevant_query_returns_useful_chunks(self):
        """A query matching the alpha-miner corpus should find BATMAN content."""
        out = _run_cli_query("alpha-miner BATMAN RTX 5060 Ti wallet workers")
        assert "BATMAN" in out or "alpha-miner" in out.lower(), \
            f"no relevant chunks found: {out[:300]}"

    def test_compact_formatter_works_on_real_query(self):
        raw = _run_cli_query("Duckets correction style", n=3, max_chars=500)
        if not COMPACT_FORMATTER.exists():
            pytest.skip("compact formatter not built yet")
        compact = _run_formatter(COMPACT_FORMATTER, raw, args=["Duckets correction style", "3", "500"])
        assert "[1]" in compact
        assert "results for:" in compact

    def test_snippet_formatter_works_on_real_query(self):
        raw = _run_cli_query("alpha-miner BATMAN RTX 5060 Ti", n=1, max_chars=500)
        if not SNIPPET_FORMATTER.exists():
            pytest.skip("snippet formatter not built yet")
        snippet = _run_formatter(SNIPPET_FORMATTER, raw, args=["x", "500"])
        assert snippet.strip(), "snippet was empty"
