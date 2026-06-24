"""
injection_scan.py — Anti-injection scanner (OWASP ASI06).

OWASP's Top 10 for Agentic AI (2026) lists "Memory and Context Poisoning"
(ASI06) as a top threat: an adversary injects text that, when retrieved
by the agent and inserted into its context, overrides the agent's
instructions.

This module is the defense. It scans text for known prompt-injection
patterns and flags suspicious content. Suspicious chunks can be:
  1. Quarantined (separated into a 'quarantine' collection)
  2. Logged with the matched pattern + severity
  3. Reviewed by a human via `quarantine_review` MCP tool

Two layers of detection:
  - Regex patterns: fast, cheap, catches known patterns
  - Heuristics: catches suspicious structure (e.g. very long
    instructions in non-prose contexts, base64 blobs, etc.)

No LLM required. This is pure-Python + regex.
Inspired by OWASP ASI06 + Microsoft "Mitigating Prompt Injection Attacks".
"""

from __future__ import annotations

import base64
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import sqlite3


DEFAULT_QUARANTINE_PATH = Path(__file__).resolve().parent.parent / "data" / "quarantine.db"


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InjectionPattern:
    id: str
    severity: int            # 1 = low (suspicious), 2 = medium (likely attack), 3 = high (definite attack)
    description: str
    pattern: re.Pattern


# Patterns are case-insensitive and try to be conservative (avoid false positives).
PATTERNS: list[InjectionPattern] = [
    # Direct instruction overrides
    InjectionPattern("ignore_previous", 3, "Tells the agent to ignore its instructions",
                     re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above|earlier|system)\s+(?:instructions?|prompts?|rules?|directives?)\b", re.IGNORECASE)),
    InjectionPattern("disregard", 3, "Tells the agent to disregard its rules",
                     re.compile(r"\bdisregard\s+(?:all\s+)?(?:previous|prior|above|earlier|system|your)\s+(?:instructions?|prompts?|rules?|directives?)\b", re.IGNORECASE)),
    InjectionPattern("forget_everything", 3, "Tells the agent to forget its context",
                     re.compile(r"\bforget\s+everything\s+(?:you\s+)?(?:know|learned|were|have)\b", re.IGNORECASE)),
    InjectionPattern("new_instructions", 3, "Injects new instructions to override system prompt",
                     re.compile(r"\b(?:new|updated|revised|override)\s+instructions?\s*[:=]", re.IGNORECASE)),
    InjectionPattern("you_are_now", 3, "Reassigns agent identity",
                     re.compile(r"\byou\s+are\s+now\s+(?:a|an|the)\s+\w+", re.IGNORECASE)),
    InjectionPattern("act_as", 2, "Tells the agent to act as something else",
                     re.compile(r"\b(?:act|behave|respond|reply)\s+as\s+(?:a|an|the|if)\s+\w+", re.IGNORECASE)),
    InjectionPattern("system_role", 3, "Tries to set a system role",
                     re.compile(r"(?:^|\n)\s*system\s*:\s*\w+", re.IGNORECASE)),
    InjectionPattern("assistant_role", 3, "Tries to set an assistant role",
                     re.compile(r"(?:^|\n)\s*assistant\s*:\s*\w+", re.IGNORECASE)),
    InjectionPattern("user_role_injection", 2, "Injects a fake user message",
                     re.compile(r"(?:^|\n)\s*user\s*:\s*(?:ignore|disregard|forget|new)", re.IGNORECASE)),
    InjectionPattern("reveal_prompt", 2, "Tries to extract the system prompt",
                     re.compile(r"\b(?:show|reveal|print|display|dump|output)\s+(?:your|the|initial)\s+(?:system\s+)?(?:prompt|instructions|rules|message)\b|\b(?:show|reveal)\s+system\s+(?:prompt|message)\b", re.IGNORECASE)),
    InjectionPattern("bypass_safety", 3, "Tries to bypass safety",
                     re.compile(r"\b(?:bypass|circumvent|disable|ignore|override)\s+(?:your\s+)?(?:safety|filter|guard|moderation|restriction|content\s+policy)\b", re.IGNORECASE)),
    InjectionPattern("jailbreak", 3, "Known jailbreak keywords",
                     re.compile(r"\b(?:DAN|jailbreak|jail\s*break|do\s+anything\s+now)\b", re.IGNORECASE)),
    InjectionPattern("developer_mode", 2, "Pretends to be 'developer mode'",
                     re.compile(r"\bdeveloper\s+mode\b", re.IGNORECASE)),
    # Data exfiltration
    InjectionPattern("exfiltrate", 3, "Tries to send data somewhere",
                     re.compile(r"\b(?:send|email|post|upload|transmit|exfiltrate)\b.{0,100}\b(?:data|files|chunks|memories|secrets|keys|tokens?|passwords?)\b.{0,40}\bto\b\s+(?:https?://|ftp://|\w+@|/\w|\w+\.\w+)", re.IGNORECASE)),
    InjectionPattern("fetch_url_injection", 2, "Tries to get the agent to fetch a URL",
                     re.compile(r"\b(?:fetch|curl|wget|visit|navigate\s+to|go\s+to|open)\s+(?:https?://|ftp://)\S+", re.IGNORECASE)),
    # Tool abuse
    InjectionPattern("run_command", 2, "Tells the agent to run a shell command",
                     re.compile(r"\b(?:run|execute|eval)\s+(?:this\s+)?(?:command|script|code)\s*[:=]?\s*[`'\"]?(?:rm\s+-rf|sudo|curl.*\|\s*(?:bash|sh)|nc\s+)", re.IGNORECASE)),
    # Hidden instructions (zero-width or unicode tricks)
    InjectionPattern("zero_width_chars", 2, "Contains zero-width unicode characters",
                     re.compile(r"[\u200b-\u200f\u2028-\u202f\u205f-\u206f\ufeff]")),
]


# ---------------------------------------------------------------------------
# Heuristics (structure-based, not regex)
# ---------------------------------------------------------------------------

@dataclass
class HeuristicHit:
    name: str
    severity: int
    description: str
    evidence: str


def heuristics_scan(text: str) -> list[HeuristicHit]:
    hits = []
    # Long uninterrupted instruction-like blocks
    if len(text) > 50:
        # Look for blocks of imperative sentences
        imperative_count = sum(1 for s in re.split(r"[.!?]\s+", text)
                                if s and s.split() and s.split()[0].lower() in {
                                    "ignore", "disregard", "forget", "reveal", "show",
                                    "send", "execute", "run", "fetch", "bypass", "override",
                                    "you", "be", "act", "respond", "reply", "do", "print",
                                })
        if imperative_count >= 3:
            hits.append(HeuristicHit(
                name="dense_imperatives",
                severity=2,
                description=f"Contains {imperative_count} imperative sentences (suspicious density)",
                evidence=text[:200],
            ))
    # Embedded base64
    b64_pattern = re.compile(r"\b([A-Za-z0-9+/]{40,}={0,2})\b")
    for m in b64_pattern.finditer(text):
        try:
            decoded = base64.b64decode(m.group(1), validate=True).decode("utf-8", errors="strict")
            if len(decoded) > 10 and any(c.isalpha() for c in decoded):
                hits.append(HeuristicHit(
                    name="embedded_base64",
                    severity=1,
                    description="Contains long base64-encoded text that decodes to readable content",
                    evidence=decoded[:100],
                ))
        except Exception:
            pass
    # Markdown code block that pretends to be instructions
    code_block = re.search(r"```[^\n]*\n([^`]*?(?:ignore|disregard|forget|reveal|send|execute|override|bypass)[^`]*?)```", text, re.IGNORECASE)
    if code_block:
        hits.append(HeuristicHit(
            name="injection_in_code_block",
            severity=2,
            description="Code block contains injection keywords",
            evidence=code_block.group(1)[:100],
        ))
    return hits


# ---------------------------------------------------------------------------
# Scan result
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    text: str
    source: Optional[str] = None
    is_clean: bool = True
    max_severity: int = 0
    pattern_hits: list[tuple[InjectionPattern, str]] = field(default_factory=list)
    heuristic_hits: list[HeuristicHit] = field(default_factory=list)
    scan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    scanned_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "scan_id": self.scan_id,
            "source": self.source,
            "is_clean": self.is_clean,
            "max_severity": self.max_severity,
            "pattern_hits": [
                {"id": p.id, "severity": p.severity, "description": p.description, "match": m}
                for p, m in self.pattern_hits
            ],
            "heuristic_hits": [
                {"name": h.name, "severity": h.severity, "description": h.description, "evidence": h.evidence}
                for h in self.heuristic_hits
            ],
            "scanned_at": self.scanned_at,
        }


# ---------------------------------------------------------------------------
# The scanner
# ---------------------------------------------------------------------------

class InjectionScanner:
    """Scan text for prompt-injection patterns.

    Default quarantine threshold: max_severity >= 2.
    Lower this if you want to flag more aggressively, raise it for fewer false positives.
    """

    def __init__(self, quarantine_threshold: int = 2):
        self.quarantine_threshold = quarantine_threshold

    def scan(self, text: str, source: Optional[str] = None) -> ScanResult:
        result = ScanResult(text=text, source=source, is_clean=True, max_severity=0)
        # Pattern scan
        for pat in PATTERNS:
            m = pat.pattern.search(text)
            if m:
                result.pattern_hits.append((pat, m.group(0)))
                result.max_severity = max(result.max_severity, pat.severity)
        # Heuristic scan
        for h in heuristics_scan(text):
            result.heuristic_hits.append(h)
            result.max_severity = max(result.max_severity, h.severity)
        result.is_clean = result.max_severity < self.quarantine_threshold
        return result

    def scan_batch(self, texts: list[str], sources: Optional[list[str]] = None
                   ) -> list[ScanResult]:
        return [
            self.scan(t, source=(sources[i] if sources else None))
            for i, t in enumerate(texts)
        ]


# ---------------------------------------------------------------------------
# Quarantine store: persist suspicious chunks for human review
# ---------------------------------------------------------------------------

SCHEMA_QUARANTINE = """
CREATE TABLE IF NOT EXISTS quarantine (
    scan_id      TEXT PRIMARY KEY,
    source       TEXT,
    text         TEXT NOT NULL,
    max_severity INTEGER NOT NULL,
    pattern_hits TEXT,        -- JSON
    heuristic_hits TEXT,      -- JSON
    is_clean     INTEGER NOT NULL,
    status       TEXT NOT NULL DEFAULT 'pending',  -- 'pending' | 'approved' | 'rejected'
    reviewer     TEXT,
    reviewed_at  REAL,
    created_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_quarantine_status ON quarantine(status);
CREATE INDEX IF NOT EXISTS idx_quarantine_severity ON quarantine(max_severity);
"""


class QuarantineStore:
    """Persist suspicious chunks for human review."""

    def __init__(self, path: str | Path = DEFAULT_QUARANTINE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA_QUARANTINE)
        self._conn.commit()

    def add(self, result: ScanResult) -> str:
        """Add a scan result to quarantine. Returns the scan_id."""
        import json
        if result.is_clean:
            raise ValueError("cannot quarantine a clean scan result")
        self._conn.execute(
            "INSERT INTO quarantine (scan_id, source, text, max_severity, pattern_hits, "
            "heuristic_hits, is_clean, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
            (result.scan_id, result.source, result.text, result.max_severity,
             json.dumps([{"id": p.id, "severity": p.severity, "match": m}
                         for p, m in result.pattern_hits]),
             json.dumps([{"name": h.name, "severity": h.severity, "evidence": h.evidence}
                         for h in result.heuristic_hits]),
             0 if result.is_clean else 1, result.scanned_at),
        )
        self._conn.commit()
        return result.scan_id

    def list_pending(self, limit: int = 100) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM quarantine WHERE status = 'pending' ORDER BY max_severity DESC, created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_all(self, status: Optional[str] = None, limit: int = 200) -> list[dict]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM quarantine WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM quarantine ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def review(self, scan_id: str, decision: str, reviewer: str = "user") -> bool:
        """Mark a scan as approved or rejected.

        decision: 'approved' (false positive, OK to ingest) or 'rejected' (true positive, drop).
        """
        if decision not in ("approved", "rejected"):
            raise ValueError(f"decision must be 'approved' or 'rejected', got {decision!r}")
        cur = self._conn.execute(
            "UPDATE quarantine SET status = ?, reviewer = ?, reviewed_at = ? "
            "WHERE scan_id = ? AND status = 'pending'",
            (decision, reviewer, time.time(), scan_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def stats(self) -> dict:
        n_total = self._conn.execute("SELECT COUNT(*) FROM quarantine").fetchone()[0]
        n_pending = self._conn.execute(
            "SELECT COUNT(*) FROM quarantine WHERE status = 'pending'"
        ).fetchone()[0]
        n_approved = self._conn.execute(
            "SELECT COUNT(*) FROM quarantine WHERE status = 'approved'"
        ).fetchone()[0]
        n_rejected = self._conn.execute(
            "SELECT COUNT(*) FROM quarantine WHERE status = 'rejected'"
        ).fetchone()[0]
        by_severity = dict(self._conn.execute(
            "SELECT max_severity, COUNT(*) FROM quarantine GROUP BY max_severity"
        ).fetchall())
        return {
            "total": n_total,
            "pending": n_pending,
            "approved": n_approved,
            "rejected": n_rejected,
            "by_severity": by_severity,
        }

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "QuarantineStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
