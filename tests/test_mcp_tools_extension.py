"""Tests for the v0.10.0 useful MCP tools extension.

Covers:
  - Brain.fsrs_review_queue  (L9)
  - Brain.decay_status       (L8)
  - Brain.forget_by_query
  - Brain.search_verbatim    (L13)
  - The new tools registered in src.mcp_server.TOOLS
  - The new tools registered in src.connectors.openclaw.TOOL_DEFINITIONS
  - The new tools registered in src/extensions/duckbot_brain/adapter.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.connectors.base import Brain
from src.memory import Memory
from src.tier import Tier
from tests._mock_embedder import MockEmbeddings


class _MockProvider:
    name = "mock"
    dim = 384

    def __init__(self, dim: int = 384):
        self._impl = MockEmbeddings(dim=dim)
        self.dim = dim

    async def embed(self, texts):
        return await self._impl.embed(texts)

    async def embed_one(self, text):
        return await self._impl.embed_one(text)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def tmp_mem():
    """Memory instance backed by a temp Chroma + mocked embedder."""
    tmp = Path(tempfile.mkdtemp(prefix="duckbot-mcp-tools-test-"))
    m = Memory(persist_dir=tmp / "chroma", embedder=_MockProvider(dim=384))
    yield m, tmp
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def brain(tmp_path):
    """Brain instance with isolated graph/blocks/quarantine paths.
    Note: brain.fsrs_review_queue and brain.decay_status internally call
    Memory(), so those tests patch the embedder separately via env var."""
    return Brain(
        graph_path=tmp_path / "graph.db",
        blocks_path=tmp_path / "blocks.db",
        quarantine_path=tmp_path / "quarantine.db",
        scan_before_remember=False,
    )


# -----------------------------------------------------------------------------
# Brain.fsrs_review_queue (L9)
# -----------------------------------------------------------------------------


def _patch_memory_factory(monkeypatch, mem):
    """Make `from src.memory import Memory` return `mem` inside base.py.

    `Brain.fsrs_review_queue` etc. import Memory lazily inside the method,
    so we patch the module attribute that base.py imports from.
    """
    import src.memory as mem_mod
    monkeypatch.setattr(mem_mod, "Memory", lambda *a, **kw: mem)


def test_fsrs_review_queue_returns_list(monkeypatch, tmp_mem):
    """fsrs_review_queue should return a list (possibly empty) and not crash."""
    mem, tmp = tmp_mem
    _patch_memory_factory(monkeypatch, mem)

    b = Brain()
    queue = b.fsrs_review_queue(k=5)
    assert isinstance(queue, list)
    # Empty store -> empty queue
    assert queue == []


def test_fsrs_review_queue_filters_by_tier(monkeypatch, tmp_mem):
    """fsrs_review_queue with tier='procedural' should not raise."""
    mem, tmp = tmp_mem
    _patch_memory_factory(monkeypatch, mem)

    b = Brain()
    q = b.fsrs_review_queue(tier="procedural", k=3)
    assert isinstance(q, list)


# -----------------------------------------------------------------------------
# Brain.decay_status (L8)
# -----------------------------------------------------------------------------


def test_decay_status_returns_summary(monkeypatch, tmp_mem):
    """decay_status returns as_of, sampled_chunks, avg_retention, by_tier."""
    mem, tmp = tmp_mem
    _patch_memory_factory(monkeypatch, mem)

    b = Brain()
    s = b.decay_status(k=20)
    assert "as_of" in s
    assert "sampled_chunks" in s
    assert "by_tier" in s
    assert isinstance(s["by_tier"], dict)


def test_decay_status_empty_store(monkeypatch, tmp_mem):
    """decay_status on an empty store: sampled_chunks=0, by_tier={}."""
    mem, tmp = tmp_mem
    _patch_memory_factory(monkeypatch, mem)

    b = Brain()
    s = b.decay_status(k=5)
    assert s["sampled_chunks"] == 0
    assert s["avg_retention"] is None


# -----------------------------------------------------------------------------
# Brain.forget_by_query
# -----------------------------------------------------------------------------


def test_forget_by_query_empty(monkeypatch, tmp_mem):
    """forget_by_query on empty store: deleted=0, deleted_ids=[]."""
    mem, tmp = tmp_mem
    _patch_memory_factory(monkeypatch, mem)

    b = Brain()
    r = b.forget_by_query("nonexistent query xyzzy", k=3)
    assert r["deleted"] == 0
    assert r["deleted_ids"] == []


@pytest.mark.asyncio
async def test_mcp_handle_recall_rejects_whitespace_query():
    from src.mcp_server import handle_recall
    import src.memory as mem_mod
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mem_mod, "Memory", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Memory should not be instantiated")))
    try:
        out = await handle_recall({"query": "   "})
    finally:
        monkeypatch.undo()
    assert "error" in out
    assert "query" in out["error"]


@pytest.mark.asyncio
async def test_mcp_handle_recall_verbatim_rejects_whitespace_query():
    from src.mcp_server import handle_recall_verbatim
    import src.connectors.base as base_mod
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(base_mod, "Brain", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Brain should not be instantiated")))
    try:
        out = await handle_recall_verbatim({"query": " \n\t "})
    finally:
        monkeypatch.undo()
    assert "error" in out
    assert "query" in out["error"]


@pytest.mark.asyncio
async def test_mcp_handle_forget_by_query_rejects_whitespace_query():
    from src.mcp_server import handle_forget_by_query
    import src.connectors.base as base_mod
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(base_mod, "Brain", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Brain should not be instantiated")))
    try:
        out = await handle_forget_by_query({"query": "  "})
    finally:
        monkeypatch.undo()
    assert "error" in out
    assert "query" in out["error"]


@pytest.mark.asyncio
async def test_mcp_handle_forget_rejects_whitespace_chunk_id():
    from src.mcp_server import handle_forget
    import src.memory as mem_mod
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mem_mod, "Memory", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Memory should not be instantiated")))
    try:
        out = await handle_forget({"chunk_id": "   "})
    finally:
        monkeypatch.undo()
    assert "error" in out
    assert "chunk_id" in out["error"]


@pytest.mark.asyncio
async def test_mcp_handle_forget_ignores_whitespace_tier(monkeypatch):
    from src.mcp_server import handle_forget
    import src.mcp_server as mcp_mod

    captured = {}

    class _FakeMemory:
        async def forget(self, chunk_id, tier=None):
            captured["chunk_id"] = chunk_id
            captured["tier"] = tier
            return True

    monkeypatch.setattr(mcp_mod, "Memory", lambda *a, **kw: _FakeMemory())
    out = await handle_forget({"chunk_id": "c1", "tier": "   "})
    assert out == {"deleted": True}
    assert captured["chunk_id"] == "c1"
    assert captured["tier"] is None


@pytest.mark.asyncio
async def test_mcp_handle_brain_inflate_rejects_whitespace_query():
    from src.mcp_server import handle_brain_inflate
    import src.memory as mem_mod
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(mem_mod, "Memory", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Memory should not be instantiated")))
    try:
        out = await handle_brain_inflate({"query": "  "})
    finally:
        monkeypatch.undo()
    assert "error" in out
    assert "query" in out["error"]


@pytest.mark.asyncio
async def test_mcp_handle_brain_skills_suggest_strips_whitespace(monkeypatch):
    from src.mcp_server import handle_brain_skills_suggest
    import src.skill_pipeline as pipeline
    captured = {}

    def fake_suggest_candidates(query, k=5, brain=None):
        captured["query"] = query
        captured["k"] = k
        return [{"chunk_id": "c1"}]

    monkeypatch.setattr(pipeline, "suggest_candidates", fake_suggest_candidates)
    out = await handle_brain_skills_suggest({"query": "  docker compose  ", "k": 2})
    assert "candidates" in out
    assert captured["query"] == "docker compose"
    assert captured["k"] == 2


@pytest.mark.asyncio
async def test_mcp_handle_brain_skills_promote_trims_whitespace(monkeypatch):
    from src.mcp_server import handle_brain_skills_promote
    import src.skill_pipeline as pipeline
    captured = {}

    def fake_promote_candidate(**kwargs):
        captured.update(kwargs)
        return {"path": "/tmp/skill.md", "slug": "skill", "chunk_id": kwargs["chunk_id"], "promoted": True}

    monkeypatch.setattr(pipeline, "promote_candidate", fake_promote_candidate)
    out = await handle_brain_skills_promote({
        "chunk_id": "  c1  ",
        "name": "  My Skill  ",
        "description": "  do the thing  ",
        "instructions": ["step"],
    })
    assert out["promoted"] is True
    assert captured["chunk_id"] == "c1"
    assert captured["name"] == "My Skill"
    assert captured["description"] == "do the thing"


# -----------------------------------------------------------------------------
# Brain.search_verbatim (L13)
# -----------------------------------------------------------------------------


def test_search_verbatim_returns_list(monkeypatch, tmp_mem):
    """search_verbatim returns a list with the right shape on empty store."""
    mem, tmp = tmp_mem
    _patch_memory_factory(monkeypatch, mem)

    b = Brain()
    matches = b.search_verbatim("nonexistent needle xyzzy")
    assert isinstance(matches, list)
    assert matches == []


def test_search_verbatim_finds_known_string(monkeypatch, tmp_path):
    """Insert a chunk with a known verbatim phrase, search should find it."""
    import asyncio
    persist = tmp_path / "chroma_sv"
    persist.mkdir()
    mem = Memory(persist_dir=persist, embedder=_MockProvider(dim=384))

    needle = "alpha-beta-gamma-delta-token-9876"
    asyncio.run(mem.remember(
        f"The secret phrase is {needle}, remember it well.",
        source_path="<test>",
        metadata={"verbatim_text": f"The secret phrase is {needle}, remember it well."},
    ))

    _patch_memory_factory(monkeypatch, mem)

    b = Brain()
    matches = b.search_verbatim(needle)
    assert len(matches) >= 1
    m = matches[0]
    assert m["match_count"] >= 1
    assert needle in m["verbatim_text"]
    # Highlights must include the needle
    all_highlight_text = "".join(h["context"] for h in m["highlights"])
    assert needle in all_highlight_text


# -----------------------------------------------------------------------------
# Tool registration: mcp_server.py
# -----------------------------------------------------------------------------


def test_mcp_server_registers_new_tools():
    """src.mcp_server.TOOLS must include the 5 new v0.10.0 tools."""
    from src.mcp_server import TOOLS
    names = {t["name"] for t in TOOLS}
    expected = {"recall_verbatim", "fsrs_review", "decay_status", "forget_by_query", "search_verbatim"}
    missing = expected - names
    assert not missing, f"Missing tools in mcp_server.TOOLS: {missing}"


def test_mcp_server_handlers_for_new_tools():
    """src.mcp_server.HANDLERS must include dispatchers for the 5 new tools."""
    from src.mcp_server import HANDLERS
    expected = {"recall_verbatim", "fsrs_review", "decay_status", "forget_by_query", "search_verbatim"}
    missing = expected - set(HANDLERS.keys())
    assert not missing, f"Missing handlers in mcp_server.HANDLERS: {missing}"


# -----------------------------------------------------------------------------
# Tool registration: connectors/openclaw.py
# -----------------------------------------------------------------------------


def test_openclaw_connector_registers_new_tools():
    """src.connectors.openclaw.TOOL_DEFINITIONS must include the 4 new v0.10.0 tools."""
    from src.connectors.openclaw import TOOL_DEFINITIONS
    names = {t["name"] for t in TOOL_DEFINITIONS}
    expected = {"brain_fsrs_review", "brain_decay_status", "brain_forget_by_query", "brain_search_verbatim"}
    missing = expected - names
    assert not missing, f"Missing tools in openclaw.TOOL_DEFINITIONS: {missing}"


def test_openclaw_connector_dispatches_new_tools():
    """src.connectors.openclaw.handle must dispatch the new tools without error."""
    from src.connectors.openclaw import handle
    # Each new tool should at least return a dict (not raise NotImplementedError)
    # We can't test full happy-path without a real store, but we can verify dispatch.
    for tool in ("brain_fsrs_review", "brain_decay_status", "brain_forget_by_query", "brain_search_verbatim"):
        r = handle(tool, {})
        assert isinstance(r, dict), f"{tool} returned non-dict: {type(r)}"
        # Tools that need args should return an error gracefully, not crash.
        # Tools with no args should return a real result.
        if tool in ("brain_fsrs_review", "brain_decay_status"):
            assert "queue" in r or "by_tier" in r or "error" in r, f"{tool} unexpected: {r}"


def test_openclaw_handle_recall_verbatim_strips_whitespace(monkeypatch):
    import src.connectors.openclaw as openclaw
    brain = MagicMock()
    brain.recall_verbatim.return_value = []
    monkeypatch.setattr(openclaw, "Brain", lambda *a, **kw: brain)
    try:
        out = openclaw.handle("brain_recall_verbatim", {"query": "   "})
    finally:
        monkeypatch.undo()
    assert "error" in out
    brain.recall_verbatim.assert_not_called()


def test_openclaw_handle_recall_strips_whitespace(monkeypatch):
    import src.connectors.openclaw as openclaw
    brain = MagicMock()
    brain.recall.return_value = []
    monkeypatch.setattr(openclaw, "Brain", lambda *a, **kw: brain)
    try:
        out = openclaw.handle("brain_recall", {"query": "   "})
    finally:
        monkeypatch.undo()
    assert "error" in out
    brain.recall.assert_not_called()


def test_openclaw_handle_recall_ignores_whitespace_tier(monkeypatch):
    import src.connectors.openclaw as openclaw
    brain = MagicMock()
    brain.recall.return_value = []
    monkeypatch.setattr(openclaw, "Brain", lambda *a, **kw: brain)
    out = openclaw.handle("brain_recall", {"query": "what changed", "tier": "   "})
    assert "error" not in out
    brain.recall.assert_called_once()
    assert brain.recall.call_args.kwargs["tier"] is None


def test_openclaw_handle_recall_rejects_invalid_tier(monkeypatch):
    import src.connectors.openclaw as openclaw
    brain = MagicMock()
    monkeypatch.setattr(openclaw, "Brain", lambda *a, **kw: brain)
    out = openclaw.handle("brain_recall", {"query": "what changed", "tier": "not-a-tier"})
    assert "error" in out
    assert "tier must be one of" in out["error"]
    brain.recall.assert_not_called()


def test_openclaw_handle_forget_by_query_strips_whitespace(monkeypatch):
    import src.connectors.openclaw as openclaw
    brain = MagicMock()
    brain.forget_by_query.return_value = {"deleted": 0, "deleted_ids": []}
    monkeypatch.setattr(openclaw, "Brain", lambda *a, **kw: brain)
    try:
        out = openclaw.handle("brain_forget_by_query", {"query": "  "})
    finally:
        monkeypatch.undo()
    assert "error" in out
    brain.forget_by_query.assert_not_called()


# -----------------------------------------------------------------------------
# Tool registration: extensions/duckbot_brain/adapter.py
# -----------------------------------------------------------------------------


def test_openclaw_extension_adapter_registers_new_tools():
    """src.extensions.duckbot_brain.adapter exposes the v0.14.0 "core agent
    surface" — the 12 tools shared with the Hermes MemoryProvider plugin.

    v0.14.0 redesign: the OpenClaw extension and the Hermes plugin now
    delegate to the same shared surface (`src.extensions.tools`), which
    intentionally excludes `brain_forget_by_query` (destructive admin
    tool, not an agent surface tool — available via the full 56-tool
    MCP server and the CLI)."""
    from src.extensions.duckbot_brain import adapter
    schemas = adapter._tool_schemas()
    names = {t["name"] for t in schemas["tools"]}
    expected = {
        "brain_wake_up", "brain_recall", "brain_recall_verbatim",
        "brain_remember", "brain_reflect", "brain_stats",
        "brain_fsrs_review", "brain_decay_status", "brain_search_verbatim",
        "brain_skills_list", "brain_skills_promote",
    }
    missing = expected - names
    assert not missing, f"Missing tools in adapter: {missing}"
    # And the destructive tool is intentionally NOT in the agent surface.
    assert "brain_forget_by_query" not in names, (
        "brain_forget_by_query is admin-tier (destructive) and must stay "
        "out of the agent surface; available via full MCP server + CLI."
    )


def test_openclaw_extension_adapter_brain_stats_uses_real_attrs(monkeypatch, brain):
    """The brain_stats tool must return fields that exist on BrainStats.

    Regression: the previous adapter code referenced s.chunks_per_tier and
    s.last_query_at which DON'T exist on BrainStats — the tool would have
    raised AttributeError if anyone called it.
    """
    # Patch _get_brain to return our test brain
    import src.extensions.duckbot_brain.adapter as adapter
    monkeypatch.setattr(adapter, "_BRAIN", brain)

    out = adapter._call_tool("brain_stats", {})
    payload = json.loads(out["content"][0]["text"])
    # These are the real fields
    assert "vector_chunks" in payload
    assert "vector_by_tier" in payload
    assert "graph_entities" in payload
    assert "quarantine_total" in payload
    # And the broken old fields are GONE
    assert "chunks_per_tier" not in payload
    assert "last_query_at" not in payload


# -----------------------------------------------------------------------------
# Regression: force_tier as a string must be accepted (not just Tier enum)
# -----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_remember_accepts_string_force_tier(tmp_mem):
    """`Memory.remember(force_tier='episodic')` must work — strings come from
    MCP / JSON-RPC callers. Regression for the bug caught while bootstrapping
    the 2026-06-23 session."""
    mem, _ = tmp_mem
    r = await mem.remember("test memory", force_tier="episodic")
    assert r.tier == Tier.EPISODIC
    assert r.stored is True


@pytest.mark.asyncio
async def test_remember_accepts_all_tier_strings(tmp_mem):
    """All four tier strings should coerce to the matching Tier enum."""
    mem, _ = tmp_mem
    for tier_str in ("working", "episodic", "semantic", "procedural"):
        r = await mem.remember(f"test for tier {tier_str}", force_tier=tier_str)
        assert r.tier.value == tier_str, f"force_tier={tier_str!r} but stored as {r.tier.value}"


class _CountingMockProvider(_MockProvider):
    """Mock embedder that records embed() batch sizes. Used to assert
    that the batched agent_facts path calls embed([...]) once with all
    facts instead of N times with one fact each."""
    def __init__(self, dim: int = 384):
        super().__init__(dim=dim)
        self.batch_sizes: list[int] = []
        self.embed_calls: int = 0

    async def embed(self, texts):
        self.embed_calls += 1
        self.batch_sizes.append(len(texts))
        return await self._impl.embed(texts)


@pytest.mark.asyncio
async def test_remember_facts_batches_embed_calls(tmp_mem, monkeypatch):
    """`Memory.remember(facts=[...])` should embed all facts in a single
    batched call instead of N separate calls. Was the source of the
    LM Studio spam when an agent passed 10+ facts per remember.

    With 5 facts: old code = 5+ separate embed() calls (1 per fact).
    New code = 1 embed() call with the whole batch.
    """
    mem, _ = tmp_mem
    # Replace the embedder with a counting one.
    counting = _CountingMockProvider()
    monkeypatch.setattr(mem, "_embedder", counting)

    facts = [
        "Duckets prefers dark mode across all UIs",
        "Decision: ChromaDB for local vector storage",
        "Rule: always run secret-scan.sh before committing",
        "Setup: use LM Studio embeddinggemma-300m for embeddings",
        "Duckets home city is Springfield",
    ]
    r = await mem.remember(
        "primary memory text longer than fifty characters to trigger bump related path",
        facts=facts,
    )
    assert r.stored

    # The primary chunk used one embed() call; the facts path should
    # have batched all 5 into ONE additional call → 2 total.
    assert counting.embed_calls <= 3, (
        f"Expected <=3 embed() calls (1 primary + 1 batched facts), "
        f"got {counting.embed_calls} with batch sizes {counting.batch_sizes}. "
        f"Old code would do 1 + 5 + N_bumps = 10+ calls."
    )
    # And one of the calls must be a batch of len(facts).
    assert len(facts) in counting.batch_sizes, (
        f"Expected one embed() call with {len(facts)} texts, "
        f"got batch sizes {counting.batch_sizes}"
    )

    # Verify all facts landed in the semantic tier.
    semantic_count = 0
    for t in facts:
        results, _ = await mem.recall(t, k=1, tier=Tier.SEMANTIC)
        if results and any(t in (r.text or "") for r in results):
            semantic_count += 1
    assert semantic_count == len(facts), (
        f"only {semantic_count}/{len(facts)} agent facts were stored"
    )


@pytest.mark.asyncio
async def test_remember_facts_dedupes_and_filters(tmp_mem, monkeypatch):
    """agent_facts shorter than 5 chars, duplicates, and over-300-char
    facts are silently dropped before the batched embed call."""
    mem, _ = tmp_mem
    counting = _CountingMockProvider()
    monkeypatch.setattr(mem, "_embedder", counting)

    facts = [
        "",                                           # empty
        "   ",                                        # whitespace
        "hi",                                         # too short
        "valid fact text",                            # ok
        "valid fact text",                            # duplicate
        "x" * 500,                                    # too long
    ]
    r = await mem.remember("primary text", facts=facts)
    assert r.stored
    # 1 primary + 1 batched (with just 1 valid fact) = 2 calls
    assert counting.embed_calls <= 2
    assert 1 in counting.batch_sizes, (
        f"Expected a batch of 1 (only the valid fact), "
        f"got {counting.batch_sizes}"
    )


@pytest.mark.asyncio
async def test_recall_rejects_empty_query(tmp_mem):
    """`Memory.recall('')` must raise ValueError instead of returning 5
    random semantically-similar chunks. Matches the MCP server's behavior."""
    mem, _ = tmp_mem
    # Add a known chunk first so the empty-query would otherwise find stuff.
    await mem.remember("the duckbot project uses cloud-only models")
    for bad in ("", "   ", "\n\n  \t"):
        with pytest.raises(ValueError, match="query must be a non-empty string"):
            await mem.recall(bad, k=5)


async def test_mcp_handle_brain_inflate_works_with_real_query():
    """Regression test for 2026-06-30 12:42 EDT end-to-end bug:
    brain_inflate raised AttributeError: 'QueryResult' object has no
    attribute 'importance'. The fix pulls importance from r.metadata
    and handles r.tier being a string OR enum.
    """
    from src.mcp_server import handle_brain_inflate
    out = await handle_brain_inflate({"query": "test", "k": 1})
    assert "error" not in out, f"brain_inflate should not error: {out.get('error')}"
    assert "context" in out
    assert isinstance(out["context"], str)
    assert len(out["context"]) > 0
