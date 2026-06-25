"""Shared pytest fixtures / autouse hooks.

Resets the Memory() singleton cache before every test so tests that
inject a fake Memory via monkeypatch.setattr see the fake, and so a
test that calls Memory() doesn't pick up a stale cached instance from a
previous test (which would leak Chroma PersistentClient handles).

See src/memory.py: _DEFAULT_MEMORY is a process-wide cache populated by
Memory.__new__ when called with no arguments. Tests that pass a fake
embedder/store via monkeypatch.setattr need the cache cleared too,
which is why the bugfixes test file does it inline. The autouse fixture
here does it for every test so individual test files don't have to.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _reset_memory_singleton():
    """Drop the cached Memory() singleton before every test."""
    from src import memory as _mem_mod
    _mem_mod._DEFAULT_MEMORY = None
    yield
    # Also clear after the test so a long-lived pytest process doesn't
    # accumulate state between test files.
    _mem_mod._DEFAULT_MEMORY = None
