"""Tests for src/skill_pipeline.py — agent-driven skill candidate pipeline.

Verifies the three design constraints:
  1. stamp_skill_candidate stores a chunk with kind='skill_candidate' (no LLM)
  2. list_candidates filters + sorts candidates correctly
  3. promote_candidate writes SKILL.md + marks promoted

Also tests the dispatch routing in src/extensions/tools.py for the
new kind='skill_candidate' remember mode + brain_skills_list/promote.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src import skill_pipeline as pipeline
from src.extensions import tools as surface


# -----------------------------------------------------------------------------
# Fixtures: fake Chroma collection + fake store
# -----------------------------------------------------------------------------

class FakeCollection:
    """Minimal stand-in for a Chroma collection used by skill_pipeline."""

    def __init__(self):
        self._chunks: dict[str, dict] = {}  # id -> {"document": str, "metadata": dict}

    def _matches_where(self, metadata: dict, where: dict | None) -> bool:
        """Minimal Chroma where-clause matching (equality only)."""
        if not where:
            return True
        for key, val in where.items():
            if key == "$and":
                if not all(self._matches_where(metadata, sub) for sub in val):
                    return False
            elif key == "$or":
                if not any(self._matches_where(metadata, sub) for sub in val):
                    return False
            else:
                if isinstance(val, dict) and "$ne" in val:
                    if metadata.get(key) == val["$ne"]:
                        return False
                elif metadata.get(key) != val:
                    return False
        return True

    def get(self, ids=None, where=None, limit=None, include=None):
        if ids is not None:
            rows = [(i, self._chunks[i]) for i in ids if i in self._chunks]
        else:
            rows = list(self._chunks.items())
        # Apply where-clause filtering (the real Chroma does this server-side).
        if where:
            rows = [(cid, d) for cid, d in rows if self._matches_where(d["metadata"], where)]
        if limit:
            rows = rows[:limit]
        if not rows:
            return {"ids": [], "documents": [], "metadatas": []}
        return {
            "ids": [r[0] for r in rows],
            "documents": [r[1]["document"] for r in rows],
            "metadatas": [r[1]["metadata"] for r in rows],
        }

    def update(self, ids, metadatas):
        for cid, md in zip(ids, metadatas):
            if cid in self._chunks:
                self._chunks[cid]["metadata"].update(md)


class FakeStore:
    def __init__(self):
        self._procedural = FakeCollection()
        self._collections = {"procedural": self._procedural}

    def collection_for(self, tier):
        # tier is a Tier enum or string
        key = tier.value if hasattr(tier, "value") else str(tier)
        return self._collections.get(key, FakeCollection())

    @property
    def procedural(self):
        return self._procedural


@pytest.fixture
def fake_procedural():
    """A FakeCollection pre-loaded with a mix of candidates + non-candidates."""
    coll = FakeCollection()
    now = time.time()
    coll._chunks = {
        "cand_old": {
            "document": "Restart the BATMAN container via docker compose",
            "metadata": {
                "kind": "skill_candidate",
                "promoted": False,
                "candidate_summary": "BATMAN restart",
                "importance": 0.7,
                "created_at": now - 3600,
                "source_path": "agent://task",
            },
        },
        "cand_new": {
            "document": "Deploy to staging using the blue-green script",
            "metadata": {
                "kind": "skill_candidate",
                "promoted": False,
                "candidate_summary": "Blue-green deploy",
                "importance": 0.8,
                "created_at": now,
                "source_path": "agent://task",
            },
        },
        "cand_promoted": {
            "document": "Fix flaky tests with pytest-timeout",
            "metadata": {
                "kind": "skill_candidate",
                "promoted": True,
                "promoted_skill_slug": "fix-flaky-tests-with-pytest-timeout",
                "candidate_summary": "Flaky test fix",
                "importance": 0.9,
                "created_at": now - 7200,
                "source_path": "agent://task",
            },
        },
        "not_a_candidate": {
            "document": "Duckets prefers dark mode",
            "metadata": {
                "kind": None,
                "importance": 0.5,
                "created_at": now,
                "source_path": "conversation://x",
            },
        },
    }
    return coll


@pytest.fixture
def reset_singletons():
    """Reset Brain + Memory singletons between tests."""
    surface.reset_brain()
    from src import memory as _mem
    _mem._DEFAULT_MEMORY = None
    yield
    surface.reset_brain()
    _mem._DEFAULT_MEMORY = None


# -----------------------------------------------------------------------------
# stamp_skill_candidate
# -----------------------------------------------------------------------------

def test_stamp_skill_candidate_calls_brain_remember_with_metadata(reset_singletons):
    """stamp_skill_candidate delegates to Brain.remember with kind + procedural tier."""
    fake_brain = MagicMock()
    fake_brain.remember.return_value = MagicMock(
        chunk_id="new_123", tier="procedural", stored=True,
    )
    result = pipeline.stamp_skill_candidate(
        text="Did a thing",
        source="agent://x",
        summary="Thing summary",
        importance=0.9,
        brain=fake_brain,
    )
    assert result.chunk_id == "new_123"
    # Verify the metadata + force_tier were passed correctly
    call_kwargs = fake_brain.remember.call_args.kwargs
    assert call_kwargs["text"] == "Did a thing"
    assert call_kwargs["source_path"] == "agent://x"
    assert call_kwargs["force_tier"] == "procedural"
    assert call_kwargs["skip_scan"] is True
    md = call_kwargs["metadata"]
    assert md["kind"] == "skill_candidate"
    assert md["promoted"] is False
    assert md["candidate_summary"] == "Thing summary"
    assert md["importance"] == 0.9


def test_stamp_skill_candidate_summary_defaults_to_truncated_text(reset_singletons):
    """If no summary given, it defaults to truncated text."""
    fake_brain = MagicMock()
    fake_brain.remember.return_value = MagicMock(chunk_id="x", tier="procedural")
    long_text = "A" * 500
    pipeline.stamp_skill_candidate(text=long_text, brain=fake_brain)
    md = fake_brain.remember.call_args.kwargs["metadata"]
    assert md["candidate_summary"] == "A" * 200


def test_stamp_skill_candidate_default_importance(reset_singletons):
    """Default importance is 0.6."""
    fake_brain = MagicMock()
    fake_brain.remember.return_value = MagicMock(chunk_id="x", tier="procedural")
    pipeline.stamp_skill_candidate(text="t", brain=fake_brain)
    assert fake_brain.remember.call_args.kwargs["metadata"]["importance"] == 0.6


# -----------------------------------------------------------------------------
# list_candidates
# -----------------------------------------------------------------------------

def test_list_candidates_excludes_promoted_by_default(fake_procedural, reset_singletons):
    """list_candidates returns only unpromoted candidates by default."""
    fake_store = FakeStore()
    fake_store._collections["procedural"] = fake_procedural

    with patch("src.skill_pipeline.Memory") as MockMem, \
         patch("src.skill_pipeline._run_async") as mock_run:
        mock_run.return_value = (fake_store, MagicMock())
        MockMem.return_value = MagicMock()
        out = pipeline.list_candidates()

    ids = [c["chunk_id"] for c in out]
    assert "cand_old" in ids
    assert "cand_new" in ids
    assert "cand_promoted" not in ids
    assert "not_a_candidate" not in ids


def test_list_candidates_includes_promoted_when_asked(fake_procedural, reset_singletons):
    fake_store = FakeStore()
    fake_store._collections["procedural"] = fake_procedural

    with patch("src.skill_pipeline.Memory") as MockMem, \
         patch("src.skill_pipeline._run_async") as mock_run:
        mock_run.return_value = (fake_store, MagicMock())
        MockMem.return_value = MagicMock()
        out = pipeline.list_candidates(include_promoted=True)

    ids = [c["chunk_id"] for c in out]
    assert "cand_promoted" in ids
    assert "not_a_candidate" not in ids


def test_list_candidates_sorted_by_recency_then_importance(fake_procedural, reset_singletons):
    """Most recent first."""
    fake_store = FakeStore()
    fake_store._collections["procedural"] = fake_procedural

    with patch("src.skill_pipeline.Memory") as MockMem, \
         patch("src.skill_pipeline._run_async") as mock_run:
        mock_run.return_value = (fake_store, MagicMock())
        MockMem.return_value = MagicMock()
        out = pipeline.list_candidates()

    # cand_new has the latest created_at
    assert out[0]["chunk_id"] == "cand_new"
    assert out[1]["chunk_id"] == "cand_old"


def test_list_candidates_empty_store(reset_singletons):
    """Empty procedural tier → empty list, no crash."""
    fake_store = FakeStore()
    with patch("src.skill_pipeline.Memory") as MockMem, \
         patch("src.skill_pipeline._run_async") as mock_run:
        mock_run.return_value = (fake_store, MagicMock())
        MockMem.return_value = MagicMock()
        out = pipeline.list_candidates()
    assert out == []


# -----------------------------------------------------------------------------
# promote_candidate
# -----------------------------------------------------------------------------

def test_promote_candidate_writes_skill_and_marks_promoted(fake_procedural, tmp_path, reset_singletons):
    """promote_candidate writes SKILL.md + marks the chunk promoted."""
    fake_store = FakeStore()
    fake_store._collections["procedural"] = fake_procedural
    skills_dir = tmp_path / "skills"

    with patch("src.skill_pipeline.Memory") as MockMem, \
         patch("src.skill_pipeline._run_async") as mock_run:
        mock_run.return_value = (fake_store, MagicMock())
        MockMem.return_value = MagicMock()
        out = pipeline.promote_candidate(
            chunk_id="cand_new",
            name="Blue-Green Deploy",
            description="Use this when deploying to staging",
            instructions=["Run the blue-green script", "Verify health checks"],
            skills_dir=skills_dir,
        )

    assert out["promoted"] is True
    assert out["chunk_id"] == "cand_new"
    assert out["slug"] == "blue-green-deploy"
    assert "path" in out

    # SKILL.md was written
    skill_path = Path(out["path"])
    assert skill_path.exists()
    content = skill_path.read_text()
    assert "blue-green-deploy" in content
    assert "Blue-Green Deploy" in content

    # The chunk metadata was updated
    md = fake_procedural._chunks["cand_new"]["metadata"]
    assert md["promoted"] is True
    assert md["promoted_skill_slug"] == "blue-green-deploy"
    assert "promoted_at" in md


def test_promote_candidate_rejects_nonexistent_chunk(fake_procedural, tmp_path, reset_singletons):
    fake_store = FakeStore()
    fake_store._collections["procedural"] = fake_procedural

    with patch("src.skill_pipeline.Memory") as MockMem, \
         patch("src.skill_pipeline._run_async") as mock_run:
        mock_run.return_value = (fake_store, MagicMock())
        MockMem.return_value = MagicMock()
        out = pipeline.promote_candidate(
            chunk_id="does_not_exist",
            name="X", description="Y", instructions=["z"],
            skills_dir=tmp_path,
        )
    assert "error" in out
    assert "does_not_exist" in out["error"]


def test_promote_candidate_rejects_non_candidate_chunk(fake_procedural, tmp_path, reset_singletons):
    """A chunk that isn't kind=skill_candidate can't be promoted."""
    fake_store = FakeStore()
    fake_store._collections["procedural"] = fake_procedural

    with patch("src.skill_pipeline.Memory") as MockMem, \
         patch("src.skill_pipeline._run_async") as mock_run:
        mock_run.return_value = (fake_store, MagicMock())
        MockMem.return_value = MagicMock()
        out = pipeline.promote_candidate(
            chunk_id="not_a_candidate",
            name="X", description="Y", instructions=["z"],
            skills_dir=tmp_path,
        )
    assert "error" in out
    assert "not a skill_candidate" in out["error"]


def test_promote_candidate_rejects_already_promoted(fake_procedural, tmp_path, reset_singletons):
    """Already-promoted chunk without overwrite → error."""
    fake_store = FakeStore()
    fake_store._collections["procedural"] = fake_procedural

    with patch("src.skill_pipeline.Memory") as MockMem, \
         patch("src.skill_pipeline._run_async") as mock_run:
        mock_run.return_value = (fake_store, MagicMock())
        MockMem.return_value = MagicMock()
        out = pipeline.promote_candidate(
            chunk_id="cand_promoted",
            name="Some Skill", description="d", instructions=["x"],
            skills_dir=tmp_path,
        )
    assert "error" in out
    assert "already promoted" in out["error"]


def test_promote_candidate_overwrite_re_promotes(fake_procedural, tmp_path, reset_singletons):
    """Already-promoted chunk WITH overwrite → re-promotes."""
    fake_store = FakeStore()
    fake_store._collections["procedural"] = fake_procedural
    skills_dir = tmp_path / "skills"

    with patch("src.skill_pipeline.Memory") as MockMem, \
         patch("src.skill_pipeline._run_async") as mock_run:
        mock_run.return_value = (fake_store, MagicMock())
        MockMem.return_value = MagicMock()
        out = pipeline.promote_candidate(
            chunk_id="cand_promoted",
            name="Fix Flaky Tests",
            description="d",
            instructions=["step"],
            skills_dir=skills_dir,
            overwrite=True,
        )
    assert out["promoted"] is True
    assert out["previously_promoted"] is True


def test_promote_candidate_requires_name_and_description(fake_procedural, reset_singletons):
    out = pipeline.promote_candidate(
        chunk_id="x", name="", description="d", instructions=["s"],
    )
    assert "error" in out


def test_promote_candidate_requires_instructions(fake_procedural, reset_singletons):
    out = pipeline.promote_candidate(
        chunk_id="x", name="n", description="d", instructions=[],
    )
    assert "error" in out
    assert "instructions" in out["error"]


# -----------------------------------------------------------------------------
# candidate_stats
# -----------------------------------------------------------------------------

def test_candidate_stats(fake_procedural, reset_singletons):
    fake_store = FakeStore()
    fake_store._collections["procedural"] = fake_procedural

    with patch("src.skill_pipeline.Memory") as MockMem, \
         patch("src.skill_pipeline._run_async") as mock_run:
        mock_run.return_value = (fake_store, MagicMock())
        MockMem.return_value = MagicMock()
        stats = pipeline.candidate_stats()

    assert stats["total"] == 3  # 3 candidates (old, new, promoted)
    assert stats["promoted"] == 1
    assert stats["unpromoted"] == 2


def test_candidate_stats_empty(reset_singletons):
    fake_store = FakeStore()
    with patch("src.skill_pipeline.Memory") as MockMem, \
         patch("src.skill_pipeline._run_async") as mock_run:
        mock_run.return_value = (fake_store, MagicMock())
        MockMem.return_value = MagicMock()
        stats = pipeline.candidate_stats()
    assert stats == {"total": 0, "promoted": 0, "unpromoted": 0}


# -----------------------------------------------------------------------------
# Dispatch routing in src/extensions/tools.py
# -----------------------------------------------------------------------------

def test_dispatch_brain_remember_skill_candidate_routes_to_stamp(reset_singletons):
    """brain_remember with kind='skill_candidate' → stamp path (blocking, returns chunk_id)."""
    fake_brain = MagicMock()
    fake_brain.remember.return_value = MagicMock(
        chunk_id="cand_dispatch_1", tier="procedural", stored=True,
    )
    surface._BRAIN = fake_brain
    out = surface.dispatch("brain_remember", {
        "text": "Learned to restart BATMAN",
        "kind": "skill_candidate",
        "summary": "BATMAN restart",
        "importance": 0.8,
    })
    assert out["status"] == "stored"
    assert out["kind"] == "skill_candidate"
    assert out["chunk_id"] == "cand_dispatch_1"
    assert out["tier"] == "procedural"
    # Verify skip_scan=True was passed (agent-authored, not user input)
    assert fake_brain.remember.call_args.kwargs["skip_scan"] is True
    assert fake_brain.remember.call_args.kwargs["force_tier"] == "procedural"


def test_dispatch_brain_remember_default_still_queues(reset_singletons):
    """brain_remember WITHOUT kind → fire-and-forget queue path (back-compat).
    Returns status=queued immediately (the daemon thread runs async)."""
    fake_brain = MagicMock()
    surface._BRAIN = fake_brain
    out = surface.dispatch("brain_remember", {"text": "normal memory"})
    assert out["status"] == "queued"
    assert out["source"] == "openclaw-extension://ad-hoc"


def test_dispatch_brain_skills_list(reset_singletons):
    """brain_skills_list dispatches to list_candidates."""
    fake_candidates = [{"chunk_id": "c1", "text": "t", "promoted": False}]
    with patch("src.skill_pipeline.list_candidates", return_value=fake_candidates):
        out = surface.dispatch("brain_skills_list", {"k": 10})
    assert "candidates" in out
    assert out["candidates"] == fake_candidates


def test_dispatch_brain_skills_promote(reset_singletons):
    """brain_skills_promote dispatches to promote_candidate."""
    expected = {"path": "/x/SKILL.md", "slug": "x", "chunk_id": "c1", "promoted": True}
    with patch("src.skill_pipeline.promote_candidate", return_value=expected) as mock_promote:
        out = surface.dispatch("brain_skills_promote", {
            "chunk_id": "c1",
            "name": "My Skill",
            "description": "desc",
            "instructions": ["step1"],
            "example": "ex",
            "emoji": "🔧",
            "overwrite": True,
        })
    assert out == expected
    mock_promote.assert_called_once_with(
        chunk_id="c1", name="My Skill", description="desc",
        instructions=["step1"], brain=surface._BRAIN,
        example="ex", emoji="🔧", overwrite=True,
    )


# -----------------------------------------------------------------------------
# Tool schema presence
# -----------------------------------------------------------------------------

def test_shared_surface_has_11_tools():
    """The shared surface now exposes 11 tools (9 + brain_skills_list + promote)."""
    names = surface.tool_names()
    assert len(names) == 11
    assert "brain_skills_list" in names
    assert "brain_skills_promote" in names
    assert "brain_remember" in names


def test_brain_remember_schema_has_kind_param():
    """brain_remember schema includes kind='skill_candidate' option."""
    schemas = surface.tool_schemas()
    remember = next(s for s in schemas if s["name"] == "brain_remember")
    props = remember["inputSchema"]["properties"]
    assert "kind" in props
    assert "skill_candidate" in props["kind"]["enum"]
    assert "summary" in props
    assert "importance" in props


def test_brain_skills_promote_schema_has_required_fields():
    schemas = surface.tool_schemas()
    promote = next(s for s in schemas if s["name"] == "brain_skills_promote")
    required = promote["inputSchema"]["required"]
    assert set(required) == {"chunk_id", "name", "description", "instructions"}


def test_function_call_schemas_includes_new_tools():
    """Hermes OpenAI function-call shape includes the new tools."""
    fcs = surface.function_call_schemas()
    names = [fc["function"]["name"] for fc in fcs]
    assert "brain_skills_list" in names
    assert "brain_skills_promote" in names
    assert len(names) == 11


def test_system_prompt_describes_agent_driven_pipeline():
    """The system prompt block explains the agent-driven flow."""
    prompt = surface.system_prompt_block()
    assert "skill_candidate" in prompt
    assert "brain_skills_promote" in prompt
    assert "agent-driven" in prompt.lower()
