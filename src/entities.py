"""
entities.py — Entity extraction (Cognee-style ECL pipeline stage 1: Extract).

Takes chunks of text and extracts entities (people, projects, files, places,
concepts, facts) and relationships between them. The extracted triples are
loaded into the temporal knowledge graph (Layer 1) by `cognify()`.

This module is **LLM-optional**:
  - By default, uses pure-Python regex patterns. No LLM call, no API cost,
    works offline. Good enough for known entities and clean prose.
  - If an LLM client is supplied, it can extract more entities from messy
    text. We support any callable that takes a list of texts and returns
    a list of (entity_name, entity_kind, related_entity, label) tuples.

Why "ECL"?
  Extract  - this module: text -> entity mentions
  Cognify  - this module: mentions -> graph triples
  Load     - this module: triples -> Graph (Layer 1)

Inspired by Cognee (https://github.com/topoteretes/cognee).
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, Optional, Protocol

from .graph import Graph


# ---------------------------------------------------------------------------
# Patterns for regex-based entity extraction
# ---------------------------------------------------------------------------

# People: "Duckets", "Ryan", "Duckets said", "the user"
PERSON_PATTERN = re.compile(
    r"\b(Duckets|Ryan|user|owner|he|she|they)\b",
    re.IGNORECASE,
)

# Projects: known project names (must be checked BEFORE person pattern)
PROJECT_NAMES = {
    "OpenClaw", "DuckBot", "DuckHive", "DuckBot RAG", "duckbot-rag-memory",
    "ai-Py-boy", "Newest Desktop Control", "NDC", "CannaAI", "Telegram",
    "AI Council", "Agent Mesh", "Cognee", "Mem0", "Letta", "Zep", "Graphiti",
    "Letta/MemGPT", "MasterDashboard", "Tavily", "MiniMax",
}

PROJECT_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(p) for p in sorted(PROJECT_NAMES, key=len, reverse=True)) + r")\b"
)

# Files: paths ending in common extensions
FILE_PATTERN = re.compile(
    r"(?:(?<=\s)|(?<=`)|(?<=\")|(?<=^))([\w./~-]+\.(?:py|js|ts|tsx|jsx|md|sh|json|yaml|yml|toml|sql|html|css))\b"
)

# Paths in OpenClaw
OPENCLAW_PATH_PATTERN = re.compile(
    r"(~/\.openclaw/[\w./-]+)"
)

# Personal facts (birthday, address, etc.)
BIRTHDAY_PATTERN = re.compile(
    r"\b(?:birthday|born on|DOB|dob)\b[:\s]+([\w/,\s]+?)(?=[.,\n)]|$)", re.IGNORECASE
)

# Relationship patterns: "X works on Y", "X created Y", "X uses Y", etc.
# We use a lookahead boundary to stop at conjunctions, prepositions, and
# punctuation without consuming them. We also optionally skip articles
# (the/a/an) so "rotated the Telegram" yields target "Telegram" not "the".
_ARTICLE = r"(?:the\s+|a\s+|an\s+)?"
_BOUNDARY = r"(?=[.,;!?]|\s+(?:and|or|but|for|with|to|from|in|on|at|by|when|that|where|while|also|his|her|their|its)\s|\s*$)"
_TGT = r"([\w./-]+(?:\s+[\w./-]+){0,2})"
RELATIONSHIP_PATTERNS = [
    (re.compile(r"(\w+)\s+(?:works on|working on|maintains|owns|created|built|wrote)\s+" + _ARTICLE + _TGT + _BOUNDARY, re.IGNORECASE), "works_on"),
    (re.compile(r"(\w+)\s+(?:uses|using)\s+" + _ARTICLE + _TGT + _BOUNDARY, re.IGNORECASE), "uses"),
    (re.compile(r"(\w+)\s+(?:depends on|requires|needs)\s+" + _ARTICLE + _TGT + _BOUNDARY, re.IGNORECASE), "depends_on"),
    (re.compile(r"(\w+)\s+(?:replaced|superseded)\s+" + _ARTICLE + _TGT + _BOUNDARY, re.IGNORECASE), "replaced"),
    (re.compile(r"(\w+)\s+(?:rotated|rotating)\s+" + _ARTICLE + _TGT + _BOUNDARY, re.IGNORECASE), "rotated"),
]


# Subjects that are not real entities (conjunctions, articles, pronouns, etc.)
SUBJECT_BLACKLIST = {"and", "or", "but", "also", "then", "so", "because", "since",
                     "although", "though", "if", "when", "while", "the", "a", "an",
                     "he", "she", "it", "they", "we", "i", "you", "his", "her", "their",
                     "this", "that", "these", "those", "there", "here"}


# ---------------------------------------------------------------------------
# Data classes for extraction results
# ---------------------------------------------------------------------------

@dataclass
class ExtractedEntity:
    name: str
    kind: str               # "person" | "project" | "file" | "place" | "concept" | "fact"
    aliases: list[str] = None  # type: ignore
    confidence: float = 1.0

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


@dataclass
class ExtractedTriple:
    source: str             # entity name (subject)
    target: str             # entity name (object)
    label: str              # predicate
    confidence: float = 1.0
    source_text: Optional[str] = None   # the chunk it came from


# ---------------------------------------------------------------------------
# The Extractor
# ---------------------------------------------------------------------------

class LLMExtractorFn(Protocol):
    """Protocol for an optional LLM-based extractor. Implementations take
    a list of texts and return a list of ExtractedTriples per text."""
    def __call__(self, texts: list[str]) -> list[list[ExtractedTriple]]: ...


class EntityExtractor:
    """Extract entities and relationships from text.

    Usage:
        extractor = EntityExtractor()  # regex-only mode
        entities, triples = extractor.extract(text, source="myfile.md")
        # Then load into graph:
        graph = Graph()
        extractor.cognify(graph, entities, triples, source="myfile.md")
    """

    def __init__(self, llm_extractor: Optional[LLMExtractorFn] = None):
        self.llm_extractor = llm_extractor

    def extract(self, text: str, source: Optional[str] = None
                ) -> tuple[list[ExtractedEntity], list[ExtractedTriple]]:
        """Extract entities and triples from a single chunk of text.

        Returns (entities, triples). Both are deduplicated within the chunk.
        """
        entities: dict[str, ExtractedEntity] = {}
        triples: list[ExtractedTriple] = []

        # ---- Projects (check FIRST so DuckBot isn't classified as person) ----
        for m in PROJECT_PATTERN.finditer(text):
            name = m.group(1)
            norm = self._normalize_name(name)
            entities[norm] = ExtractedEntity(name=norm, kind="project")

        # ---- People ----
        for m in PERSON_PATTERN.finditer(text):
            name = m.group(1)
            if name.lower() in ("user", "owner", "he", "she", "they"):
                # Skip pronouns/role words; they're not real entities
                continue
            norm = self._normalize_name(name)
            # Don't override a project classification
            if norm not in entities:
                entities[norm] = ExtractedEntity(name=norm, kind="person")

        # ---- Files ----
        for m in FILE_PATTERN.finditer(text):
            path = m.group(1)
            entities.setdefault(path, ExtractedEntity(name=path, kind="file"))

        # ---- OpenClaw paths ----
        for m in OPENCLAW_PATH_PATTERN.finditer(text):
            path = m.group(1)
            entities.setdefault(path, ExtractedEntity(name=path, kind="file"))

        # ---- Personal facts ----
        for m in BIRTHDAY_PATTERN.finditer(text):
            when = m.group(1).strip().rstrip(",").strip()
            if not when:
                continue
            entities.setdefault("Duckets birthday", ExtractedEntity(
                name="Duckets birthday", kind="fact",
                aliases=[f"birthday:{when}"]
            ))

        # ---- Relationship patterns ----
        for pattern, label in RELATIONSHIP_PATTERNS:
            for m in pattern.finditer(text):
                src, tgt = m.group(1), m.group(2).strip().rstrip(".,;")
                if not src or not tgt or len(tgt) < 2 or len(src) < 2:
                    continue
                src_n = self._normalize_name(src)
                tgt_n = self._normalize_name(tgt)
                # Filter out junk matches
                if src_n.lower() in SUBJECT_BLACKLIST:
                    continue
                if tgt_n.lower() in SUBJECT_BLACKLIST or tgt_n.lower() in ("it", "this", "that"):
                    continue
                # If target is a known project, ensure it's in entities.
                # PROJECT_NAMES is mixed-case ("OpenClaw", "DuckBot"); compare
                # case-insensitively so lowercase variants in prose still match.
                if tgt_n.lower() in {p.lower() for p in PROJECT_NAMES}:
                    entities.setdefault(tgt_n, ExtractedEntity(name=tgt_n, kind="project"))
                triples.append(ExtractedTriple(
                    source=src_n, target=tgt_n, label=label,
                    source_text=text[:200] if source else None,
                ))

        # ---- LLM pass (optional) ----
        if self.llm_extractor is not None:
            try:
                llm_results = self.llm_extractor([text])
                if llm_results:
                    for t in llm_results[0]:
                        triples.append(t)
                        # Make sure endpoints are in entities
                        entities.setdefault(
                            self._normalize_name(t.source),
                            ExtractedEntity(name=self._normalize_name(t.source), kind="concept"),
                        )
                        entities.setdefault(
                            self._normalize_name(t.target),
                            ExtractedEntity(name=self._normalize_name(t.target), kind="concept"),
                        )
            except Exception:
                # LLM failures are non-fatal; regex output stands
                pass

        return list(entities.values()), triples

    def extract_batch(self, texts: list[str], sources: Optional[list[str]] = None
                      ) -> tuple[list[ExtractedEntity], list[ExtractedTriple]]:
        """Extract from many chunks at once. Returns the global deduped sets."""
        all_entities: dict[str, ExtractedEntity] = {}
        all_triples: list[ExtractedTriple] = []
        for i, t in enumerate(texts):
            src = sources[i] if sources else None
            ents, trips = self.extract(t, source=src)
            for e in ents:
                # Merge aliases if same name seen
                if e.name in all_entities:
                    existing = all_entities[e.name]
                    existing.aliases = list(set(existing.aliases) | set(e.aliases))
                else:
                    all_entities[e.name] = e
            all_triples.extend(trips)
        return list(all_entities.values()), all_triples

    def cognify(self, graph: Graph,
                entities: Iterable[ExtractedEntity],
                triples: Iterable[ExtractedTriple],
                source: Optional[str] = None,
                at: Optional[float] = None) -> dict:
        """Load extracted entities and triples into the graph.

        Idempotent: re-running with the same data is a no-op.
        Returns counts: {'entities_added', 'triples_added', 'triples_deduped'}.
        """
        if at is None:
            at = time.time()
        ents_added = 0
        trips_added = 0
        trips_dedup = 0
        for e in entities:
            existing = graph.find_entity(e.name)
            if existing is None:
                graph.upsert_entity(name=e.name, kind=e.kind, aliases=e.aliases)
                ents_added += 1
        # Group triples by (source, target, label) to dedupe
        seen: set[tuple[str, str, str]] = set()
        for t in triples:
            key = (t.source, t.target, t.label)
            if key in seen:
                trips_dedup += 1
                continue
            seen.add(key)
            src_ent = graph.find_entity(t.source)
            tgt_ent = graph.find_entity(t.target)
            if src_ent is None:
                # Auto-create as concept if unknown
                src_ent = graph.upsert_entity(name=t.source, kind="concept")
                ents_added += 1
            if tgt_ent is None:
                tgt_ent = graph.upsert_entity(name=t.target, kind="concept")
                ents_added += 1
            graph.add_relationship(
                src_ent.id, tgt_ent.id, t.label,
                valid_from=at, source=source, confidence=t.confidence,
            )
            trips_added += 1
        return {
            "entities_added": ents_added,
            "triples_added": trips_added,
            "triples_deduped": trips_dedup,
        }

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize an entity name for consistent matching."""
        n = name.strip()
        # Lowercase pronouns/role words out (handled by caller)
        # Trim trailing punctuation
        n = n.rstrip(".,;:!?")
        # Collapse whitespace
        n = re.sub(r"\s+", " ", n)
        return n


# ---------------------------------------------------------------------------
# Convenience: scan a markdown file
# ---------------------------------------------------------------------------

def extract_from_markdown_file(path: str) -> tuple[list[ExtractedEntity], list[ExtractedTriple]]:
    """Read a markdown file and extract entities/triples. Pure I/O wrapper."""
    from pathlib import Path
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    extractor = EntityExtractor()
    return extractor.extract(text, source=path)
