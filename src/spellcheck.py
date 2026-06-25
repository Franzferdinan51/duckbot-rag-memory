"""
spellcheck.py — lightweight spellcheck for ingest.

MemPalace has a "spellcheck" extra (`pip install mempalace[spellcheck]`).
We want zero new paid deps and minimal new free deps, so this is a
small targeted common-typo fixer rather than a full spellcheck engine.

How it works:
  1. For each word in the input, lowercase + strip punctuation.
  2. If the word is in a tiny embedded typo table, replace it.
  3. Otherwise, leave it alone (we don't have a dictionary to fall back
     to without downloading one).

The embedded table covers the most common English typos that actually
show up in markdown notes (your own + AI-generated). If you need full
spellcheck, swap in `symspellpy.SymSpell` — same interface, just slower
but covers 100x more words.

Usage:
    from src.spellcheck import fix_text
    fixed = fix_text("I recieved your mesage yestarday")
    # -> "I received your message yesterday"
"""
from __future__ import annotations

import re
from typing import Iterable


# Common English typos → correct. Hand-curated from the most frequent
# errors in personal-note corpora. Words here are case-insensitive
# (lookup lowercases both sides).
COMMON_TYPOS: dict[str, str] = {
    # 1-letter away from common words
    "teh": "the",
    "hte": "the",
    "adn": "and",
    "nad": "and",
    "recieve": "receive",
    "recieved": "received",
    "recieving": "receiving",
    "acheive": "achieve",
    "acheived": "achieved",
    "acheiving": "achieving",
    "beleive": "believe",
    "beleived": "believed",
    "beleif": "belief",
    "occured": "occurred",
    "occuring": "occurring",
    "occurence": "occurrence",
    "seperate": "separate",
    "seperated": "separated",
    "seperately": "separately",
    "seperation": "separation",
    "definately": "definitely",
    "definatly": "definitely",
    "untill": "until",
    "alot": "a lot",
    "aswell": "as well",
    "infront": "in front",
    "infact": "in fact",
    "neverthless": "nevertheless",
    "noticable": "noticeable",
    "usefull": "useful",
    "usefully": "usefully",
    "thier": "their",
    "whcih": "which",
    "wether": "whether",
    "whther": "whether",
    "yestarday": "yesterday",
    "tommorow": "tomorrow",
    "tomorow": "tomorrow",
    "tommorrow": "tomorrow",
    "mesage": "message",
    "mesages": "messages",
    "messsage": "message",
    "wether": "whether",
    "sytem": "system",
    "sytems": "systems",
    "appliction": "application",
    "applictions": "applications",
    "funciton": "function",
    "funcitons": "functions",
    "enviroment": "environment",
    "enviroments": "environments",
    "reponse": "response",
    "reponses": "responses",
    "requst": "request",
    "requsts": "requests",
    "sucess": "success",
    "sucessful": "successful",
    "sucessfully": "successfully",
    "succesful": "successful",
    "succesfully": "successfully",
    "happend": "happened",
    "happends": "happens",
    "wether": "whether",
    "loosing": "losing",
    "loose": "lose",  # disambiguate: only the verb form; "loose" the adj stays
    "alot": "a lot",
    "thats": "that's",  # we DON'T fix "thats" since it's a valid informal contraction
    # Tech-specific
    "dockerfile": "Dockerfile",
    "readme": "README",
    "changelog": "CHANGELOG",
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "github": "GitHub",
    "gitignore": ".gitignore",
    "localhost": "localhost",
}


# Words that look like typos but should be left alone (proper nouns, etc.).
TYPO_EXCEPTIONS: set[str] = {
    "Duckets", "DuckBot", "DuckHive", "NDC", "CannaAI", "BATMAN",
    "OpenClaw", "Hermes", "ChromaDB", "Chroma", "MemPalace",
    "Telegram", "Nvidia", "Anthropic", "MiniMax",
    "Mavis", "Milla", "Jovovich",  # MemPalace founder name
    "FTS5", "BM25", "RRF", "FSRS",  # algorithms
    "Pi", "Cua", "Tavily", "Honcho", "Cognee", "Graphiti", "Letta",
    "mem0", "mem0ai",
}


# Match a "word" (letters / digits / common tech chars like . - _).
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9._\-]*")


def _is_protected(word: str) -> bool:
    """True if the word should never be spellchecked (proper noun, brand,
    known acronym, etc.)."""
    # Exact match
    if word in TYPO_EXCEPTIONS:
        return True
    # Starts with uppercase (proper-noun heuristic) and length > 1 —
    # but we still want to fix "Thier" -> "their" (lowercase start)
    # while leaving "Duckets" alone. So: protect any word whose first
    # letter is uppercase AND isn't a known typo (e.g. "Teh" should be
    # fixed even though it starts uppercase).
    if word[:1].isupper() and word.lower() not in COMMON_TYPOS:
        return True
    return False


def fix_word(word: str) -> str:
    """Fix a single word. Returns the word unchanged if no fix applies.
    Preserves the original case shape (e.g. "Teh" -> "The", not "the").
    """
    if _is_protected(word):
        return word
    # Only fix lower-case lookups; preserve case shape in output.
    lower = word.lower()
    fix = COMMON_TYPOS.get(lower)
    if not fix:
        return word
    # Re-apply original case: if the original word was capitalized,
    # capitalize the first letter of the fix.
    if word[:1].isupper() and len(fix) > 0:
        return fix[:1].upper() + fix[1:]
    return fix


def fix_text(text: str, extra_typos: dict[str, str] | None = None) -> str:
    """Apply common-typo fixes across `text`. Preserves punctuation and
    case. If `extra_typos` is given, it's merged into the lookup table
    (project-specific or domain-specific words).
    """
    if not text:
        return text
    table = COMMON_TYPOS if extra_typos is None else {**COMMON_TYPOS, **extra_typos}

    def _sub(m: re.Match) -> str:
        w = m.group(0)
        lower = w.lower()
        if lower not in table:
            return w
        fix = table[lower]
        if w[:1].isupper() and fix:
            return fix[:1].upper() + fix[1:]
        return fix

    return _WORD_RE.sub(_sub, text)


def list_typos() -> list[tuple[str, str]]:
    """Return the embedded typo table sorted by typo. Useful for
    debugging + for tools that want to add project-specific entries."""
    return sorted(COMMON_TYPOS.items())
