"""
session_ingest.py — unified session ingest (Phase F, MemHub-style).

The fleet's agents each keep their own session stores; the brain only
ever saw two markdown dirs. This module normalizes the two big local
session sources into plain conversation text and runs the standard
distillation pipeline over them (Hy-Memory writer: facts with Phase A
reconciliation, Phase B intentions, Phase D episodes):

  grok-local   ~/.grok-local/sessions/<project>/<session>/chat_history.jsonl
               records {type: system|user|assistant|..., content: str|parts}
  zcode-local  ~/.zcode-local/cli/rollout/model-io-sess_*.jsonl
               per-call records; the LAST record's request.messages holds
               the full history, response.text the final reply.

Boilerplate (system prompts, injected <user_info>/<rules> wrappers,
system-reminders) is stripped during normalization — it is not memory.
Ingest is state-tracked (path -> mtime in a small JSON file) so reruns
only process changed sessions; remember()'s content-hash ids make even
an untracked re-ingest a no-op at the store level. Model-required and
fail-open like every writer pass in this codebase.
Import as src.session_ingest.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_WRAPPER_RE = re.compile(
    r"<(user_info|rules|system-reminder)>.*?</\1>", re.DOTALL)
_QUERY_RE = re.compile(r"<user_query>(.*?)</user_query>", re.DOTALL)
_TAG_FRAGMENT_RE = re.compile(
    r"\[escaped: (?:system-reminder|user_info|rules)>|</?(?:system-reminder|user_info|rules)>")


def _clean_user_text(text: str) -> str:
    m = _QUERY_RE.search(text)
    if m:
        return m.group(1).strip()
    return _TAG_FRAGMENT_RE.sub("", _WRAPPER_RE.sub("", text)).strip()


def _parts_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                t = part.get("text")
                if isinstance(t, str):
                    out.append(t)
            elif isinstance(part, str):
                out.append(part)
        return "\n".join(out)
    return ""


def grok_session_text(path: str | Path) -> str:
    """Normalize a grok-local chat_history.jsonl into conversation text."""
    turns: list[str] = []
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return ""
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        role = rec.get("type")
        if role not in ("user", "assistant"):
            continue
        text = _parts_text(rec.get("content")).strip()
        if not text:
            continue
        if role == "user":
            text = _clean_user_text(text)
            if not text:
                continue
        turns.append(f"{role.upper()}: {text}")
    return "\n\n".join(turns)


def zcode_rollout_text(path: str | Path) -> str:
    """Normalize a zcode-local rollout jsonl into conversation text."""
    last = None
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return ""
    for line in lines:
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and isinstance(rec.get("request"), dict):
            last = rec
    if last is None:
        return ""
    turns: list[str] = []
    for msg in last["request"].get("messages") or []:
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        text = _parts_text(msg.get("content")).strip()
        if text:
            turns.append(f"{role.upper()}: {text}")
    resp_text = ((last.get("response") or {}).get("text") or "").strip()
    if resp_text:
        turns.append(f"ASSISTANT: {resp_text}")
    return "\n\n".join(turns)


def discover_sessions(kind: str, *, days: Optional[float] = None,
                      home: Optional[Path] = None) -> list[Path]:
    """Find session files for a source kind, newest first."""
    home = home or Path.home()
    if kind == "grok":
        files = sorted(home.glob(".grok-local/sessions/*/*/chat_history.jsonl"))
    elif kind == "zcode":
        files = [p for p in home.glob(".zcode-local/cli/rollout/model-io-sess_*.jsonl")
                 if "no-session" not in p.name]
    else:
        return []
    if days is not None:
        import time
        cutoff = time.time() - days * 86400
        files = [p for p in files if p.stat().st_mtime >= cutoff]
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)


_TEXT_FOR_KIND = {"grok": grok_session_text, "zcode": zcode_rollout_text}


def _segments(text: str, *, max_chars: int = 6000) -> list[str]:
    """Split conversation text into <=max_chars segments on turn breaks."""
    segs: list[str] = []
    cur = ""
    for turn in text.split("\n\n"):
        if cur and len(cur) + len(turn) + 2 > max_chars:
            segs.append(cur)
            cur = turn
        else:
            cur = f"{cur}\n\n{turn}" if cur else turn
    if cur.strip():
        segs.append(cur)
    return segs


def _load_state(state_path: Path) -> dict:
    try:
        return json.loads(Path(state_path).read_text())
    except (OSError, ValueError):
        return {}


async def ingest_files(mem, files: list[tuple[str, Path]], *,
                       model: Optional[str] = None,
                       state_path: Optional[Path] = None) -> dict:
    """Distill session files into the bank. Returns per-run counts.

    files: [(kind, path)]. Unchanged files (same mtime in state) skip.
    Without a model nothing is stored (the writer IS the pipeline).
    """
    from . import hy_memory
    from .experience import mine_episodes
    from .layers import extract_intention_layer

    counts = {"files": 0, "skipped": 0, "facts": 0, "episodes": 0,
              "intentions": 0}
    if not model:
        return counts
    state = _load_state(state_path) if state_path else {}
    dirty = False
    for kind, path in files:
        key = str(path)
        mtime = path.stat().st_mtime if path.exists() else 0
        if state.get(key) == mtime:
            counts["skipped"] += 1
            continue
        text = _TEXT_FOR_KIND[kind](path)
        if not text.strip():
            state[key] = mtime
            dirty = True
            continue
        counts["files"] += 1
        for seg in _segments(text):
            for fact in hy_memory.extract_memories_hy(seg, model=model):
                await mem.remember(fact, source_path=key,
                                   force_tier="semantic",
                                   reconcile_model=model)
                counts["facts"] += 1
            eps = await mine_episodes(mem, seg, model=model,
                                      source_path=key)
            counts["episodes"] += len(eps)
            intents = await extract_intention_layer(
                mem, [seg], model=model, source_path=key)
            counts["intentions"] += len(intents)
        state[key] = mtime
        dirty = True
    if state_path and dirty:
        Path(state_path).write_text(json.dumps(state, indent=1))
    return counts
