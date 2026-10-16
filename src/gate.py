"""
gate.py — optional D1/SystemOne gatekeeper for memory writes (Phase 3).

Before a chunk reaches the (RAM-heavy) writer model, this gate asks
Duckets' D1 decision layer to classify it in ONE 4-way choice:

    chat | working | code | none

served by LM Studio's POST /v1/systemone with model d1-3b — the exact
wire dialect systemone/d1.py uses ({"model", "state", "questions"} with
per-question {"type", "instructions", "criteria"}). The shim's
/v1/decisions route is NOT used: it is served by the GLiClass fallback,
whose store/skip probabilities do not discriminate (verified live
2026-10-16: junk scored store 0.66). A single bank-or-none choice
calibrates far better on d1-3b than a binary store/skip question
(verified on calibration chunks: bio->chat 0.92, junk->none 0.84,
project-state->working 0.97).

Safety posture:
- verdict="skip" only when "none" wins with probability >= SKIP_THRESHOLD.
  A mushy "none" stores into the best real bank instead: a false store
  costs a junk row; a false skip costs a lost memory.
- Fail-open: any transport error or unparseable answer returns
  verdict="store", gated=False (the d1.py philosophy).
- The decision model defaults to d1-3b, the sanctioned decider.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

DECISION_MODEL = "d1-3b"
BANKS = ("chat", "working", "code")
SKIP_THRESHOLD = 0.60

_CRITERIA = {
    "chat": "facts about the user, family, life, or preferences",
    "working": "projects, machines, training runs, or current work state",
    "code": "programming knowledge or technical decisions",
    "none": "no lasting value: small talk, samples, placeholders, boilerplate",
}
_INSTRUCTIONS = "Classify this text for a personal assistant's long-term memory."


@dataclass
class GateDecision:
    verdict: str = "store"          # "store" | "skip"
    bank: str = "chat"              # one of BANKS
    confidence: float = 0.0         # probability of the winning option
    gated: bool = False             # False => fail-open pass
    raw: dict = field(default_factory=dict)


def _d1_url() -> str:
    raw = (
        os.environ.get("DUCKBOT_D1_URL", "").strip()
        or os.environ.get("LMSTUDIO_URL", "").strip()
        or "http://127.0.0.1:1234"
    )
    return raw.rstrip("/") + "/v1/systemone"


def _post_systemone(payload: dict, timeout: float) -> Optional[dict]:
    try:
        import httpx
    except ImportError:
        return None
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(_d1_url(), json=payload)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def _winning_option(answer, option_names: list[str]) -> tuple[Optional[str], float]:
    """Winning option + probability from a D1 answer, shape-tolerant."""
    if not isinstance(answer, dict):
        return None, 0.0
    probs = answer.get("probabilities") or answer.get("distribution") or {}
    if isinstance(probs, dict) and probs:
        best_name, best_p = None, -1.0
        for key, val in probs.items():
            try:
                p = float(val)
            except (TypeError, ValueError):
                continue
            name = None
            if key in option_names:
                name = key
            elif isinstance(key, str) and len(key) == 1 and key.isalpha():
                idx = ord(key.upper()) - ord("A")
                if 0 <= idx < len(option_names):
                    name = option_names[idx]
            if name is not None and p > best_p:
                best_name, best_p = name, p
        if best_name is not None:
            return best_name, best_p
    for key in ("choice", "label", "answer", "value"):
        v = answer.get(key)
        if isinstance(v, str) and v in option_names:
            return v, float(answer.get("confidence", 0.0) or 0.0)
    return None, 0.0


def gate_chunk(
    chunk_text: str,
    *,
    model: str = DECISION_MODEL,
    timeout: float = 30.0,
) -> GateDecision:
    """Classify a chunk via D1: skip it, or store it into a bank."""
    if not chunk_text.strip():
        return GateDecision(verdict="skip", bank="chat", confidence=1.0, gated=True)
    options = list(BANKS) + ["none"]
    payload = {
        "model": model,
        "state": chunk_text[:4000],
        "questions": {
            "bank": {
                "type": "choice",
                "instructions": _INSTRUCTIONS,
                "criteria": _CRITERIA,
            }
        },
    }
    data = _post_systemone(payload, timeout)
    if not data:
        return GateDecision()  # fail-open
    answers = data.get("answers") or {}
    winner, conf = _winning_option(answers.get("bank"), options)
    if winner is None:
        return GateDecision(raw=data)  # unparseable => fail-open
    if winner == "none":
        if conf >= SKIP_THRESHOLD:
            return GateDecision(verdict="skip", bank="chat", confidence=conf,
                                gated=True, raw=data)
        return GateDecision(verdict="store", bank="chat", confidence=conf,
                            gated=True, raw=data)
    return GateDecision(verdict="store", bank=winner, confidence=conf,
                        gated=True, raw=data)
