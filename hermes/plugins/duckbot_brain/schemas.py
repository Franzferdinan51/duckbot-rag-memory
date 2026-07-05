"""schemas.py - Pydantic input/output schemas for DuckBot brain tools."""
# MIT License - see LICENSE in the repository root.
from __future__ import annotations
from typing import Literal, Optional

try:
    from pydantic import BaseModel, Field
    _PYDANTIC_AVAILABLE = True
except Exception:
    _PYDANTIC_AVAILABLE = False
    BaseModel = object  # type: ignore[assignment,misc]

def _lazy_schemas() -> list[dict]:
    try:
        from src.extensions.tools import TOOLS
        return TOOLS
    except Exception:
        return []

class BrainWakeUpInput(BaseModel if _PYDANTIC_AVAILABLE else dict):
    if _PYDANTIC_AVAILABLE:
        query: Optional[str] = Field(default=None, description='optional anchor')
        k: int = Field(default=8, description='max memories to return')
        include_blocks: bool = Field(default=True)
        include_graph: bool = Field(default=True)
        include_fsrs_review: bool = Field(default=True)

class BrainRecallInput(BaseModel if _PYDANTIC_AVAILABLE else dict):
    if _PYDANTIC_AVAILABLE:
        query: str = Field(description='search query')
        k: int = Field(default=5)
        tier: Optional[Literal['working','episodic','semantic','procedural']] = Field(default=None)
        min_importance: Optional[float] = Field(default=None, ge=0.0, le=1.0)
        rerank: bool = Field(default=False)
        decay: bool = Field(default=False)
        tier_priors: bool = Field(default=False)
        tier_priors_overrides: Optional[dict] = Field(default=None)
        fsrs: bool = Field(default=False)

class BrainRememberInput(BaseModel if _PYDANTIC_AVAILABLE else dict):
    if _PYDANTIC_AVAILABLE:
        text: str = Field(description='the memory content')
        source: str = Field(default='openclaw-extension://ad-hoc', description='where this came from')
        tier: Optional[Literal['working','episodic','semantic','procedural']] = Field(default=None)
        kind: Optional[Literal['skill_candidate']] = Field(default=None)
        summary: Optional[str] = Field(default=None)
        importance: float = Field(default=0.6, ge=0.0, le=1.0)
        trust_level: Literal['full','standard'] = Field(default='full')
        facts: Optional[list[str]] = Field(default=None)

class BrainReflectInput(BaseModel if _PYDANTIC_AVAILABLE else dict):
    if _PYDANTIC_AVAILABLE:
        lookback_days: int = Field(default=7, ge=1)
        max_chunks: int = Field(default=200, ge=1)

class BrainStatsInput(BaseModel if _PYDANTIC_AVAILABLE else dict):
    pass

class BrainFsrsReviewInput(BaseModel if _PYDANTIC_AVAILABLE else dict):
    if _PYDANTIC_AVAILABLE:
        tier: Optional[Literal['working','episodic','semantic','procedural']] = Field(default=None)
        k: int = Field(default=10, ge=1)

class BrainDecayStatusInput(BaseModel if _PYDANTIC_AVAILABLE else dict):
    if _PYDANTIC_AVAILABLE:
        tier: Optional[Literal['working','episodic','semantic','procedural']] = Field(default=None)
        k: int = Field(default=50, ge=1)

class BrainSearchVerbatimInput(BaseModel if _PYDANTIC_AVAILABLE else dict):
    if _PYDANTIC_AVAILABLE:
        needle: str = Field(description='exact substring to search for')
        k: int = Field(default=5, ge=1)

class BrainSkillsListInput(BaseModel if _PYDANTIC_AVAILABLE else dict):
    if _PYDANTIC_AVAILABLE:
        include_promoted: bool = Field(default=False)
        k: int = Field(default=50, ge=1)

class BrainSkillsSuggestInput(BaseModel if _PYDANTIC_AVAILABLE else dict):
    if _PYDANTIC_AVAILABLE:
        query: str = Field(description='semantic anchor')
        k: int = Field(default=5, ge=1)

class BrainSkillsPromoteInput(BaseModel if _PYDANTIC_AVAILABLE else dict):
    if _PYDANTIC_AVAILABLE:
        chunk_id: str = Field(description='the candidate chunk to promote')
        name: str = Field(description='human-readable skill name')
        description: str = Field(description='one-line trigger phrase')
        instructions: Optional[list[str]] = Field(default=None)
        instructions_markdown: Optional[str] = Field(default=None)
        example: Optional[str] = Field(default=None)
        emoji: Optional[str] = Field(default=None)
        overwrite: bool = Field(default=False)

def get_tool_schemas() -> list[dict]:
    return _lazy_schemas()

__all__ = [
    'BrainWakeUpInput', 'BrainRecallInput', 'BrainRememberInput',
    'BrainReflectInput', 'BrainStatsInput', 'BrainFsrsReviewInput',
    'BrainDecayStatusInput', 'BrainSearchVerbatimInput',
    'BrainSkillsListInput', 'BrainSkillsSuggestInput', 'BrainSkillsPromoteInput',
    'get_tool_schemas',
]
