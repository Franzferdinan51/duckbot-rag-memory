"""
coding.py — coding memory with version-evolution diff-chains (Phase E).

Hy's Coding scene: an agent that works in a repo across sessions should
remember WHY the code changed, not just re-read it. This module walks a
repo's git history, distills each commit into a decision record via the
Hy-Memory writer, and stores the records as EPISODIC chunks with
layer="coding" metadata {repo, commit, files, chain_prev} — where
chain_prev links a commit to the most recent earlier commit that touched
an overlapping file, forming the revision chain per file/area.

Idempotent: commits already in the bank are skipped. Fail-open: a
commit the writer can't distill is stored as its raw subject line
rather than dropped, so the chain never has silent gaps.
Import as src.coding.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_RS = "\x1e"  # record separator
_FS = "\x1f"  # field separator


def iter_commits(repo_path: str | Path, *, limit: int = 200) -> list[dict]:
    """Read a repo's history (oldest first) via git log. [] on any error."""
    repo = Path(repo_path).expanduser()
    if not (repo / ".git").exists():
        return []
    fmt = _RS + _FS.join(["%H", "%at", "%s", "%b"])
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "log", "--reverse", "--name-only",
             f"--format={fmt}", f"-n{limit}"],
            capture_output=True, text=True, timeout=120)
    except Exception as exc:
        logger.warning("git log failed for %s (%s)", repo, exc)
        return []
    if out.returncode != 0:
        return []
    commits: list[dict] = []
    for rec in out.stdout.split(_RS):
        rec = rec.strip("\n")
        if not rec.strip():
            continue
        parts = rec.split(_FS)
        if len(parts) < 4:
            continue
        head, body_and_files = parts[:3], parts[3]
        lines = [ln.strip() for ln in body_and_files.splitlines()]
        body = "\n".join(lines[:-0] if False else lines).strip()
        files = [ln for ln in lines if "/" in ln or "." in ln.split("/")[-1]]
        # Body vs files: git emits body, a blank line, then file names.
        # Split on the first blank-line run instead of guessing by shape.
        raw = body_and_files
        if "\n\n" in raw:
            body_part, _, files_part = raw.partition("\n\n")
            body = body_part.strip()
            files = [ln.strip() for ln in files_part.splitlines() if ln.strip()]
        commits.append({
            "hash": head[0].strip(),
            "ts": int(head[1]) if head[1].strip().isdigit() else 0,
            "subject": head[2].strip(),
            "body": body,
            "files": files,
        })
    return commits


def _stored_commit_hashes(mem, repo_name: str, store=None) -> set[str]:
    coll = (store or mem._store).collection_for(_tier())
    got = coll.get(include=["metadatas"])
    hashes: set[str] = set()
    for meta in got.get("metadatas") or []:
        meta = meta or {}
        if meta.get("layer") == "coding" and meta.get("repo") == repo_name:
            if meta.get("commit"):
                hashes.add(meta["commit"])
    return hashes


def _tier():
    from .tier import Tier
    return Tier.EPISODIC


async def store_chain(mem, repo_path: str | Path, *,
                      model: Optional[str] = None,
                      limit: int = 200) -> dict:
    """Distill + store a repo's commit chain. Returns counts."""
    from . import hy_memory

    repo = Path(repo_path).expanduser()
    repo_name = repo.name
    commits = iter_commits(repo, limit=limit)
    store, _embedder = await mem._ensure_initialized()
    already = _stored_commit_hashes(mem, repo_name, store=store)
    file_last: dict[str, str] = {}
    # Seed the chain map from history order (all commits, stored or not)
    stored = skipped = fallback = 0
    for c in commits:
        prev = None
        for f in c["files"]:
            if f in file_last:
                prev = file_last[f]
                break
        for f in c["files"]:
            file_last[f] = c["hash"]
        if c["hash"] in already:
            skipped += 1
            continue
        decision = None
        if model:
            decision = hy_memory.extract_commit_decision(
                c["subject"], c["body"], c["files"], model=model)
        if not decision:
            decision = c["subject"]
            fallback += 1
        meta = {"layer": "coding", "repo": repo_name, "commit": c["hash"],
                "files": ",".join(c["files"][:10])}
        if prev:
            meta["chain_prev"] = prev
        await mem.remember(decision, source_path=f"git:{repo_name}",
                           force_tier="episodic", metadata=meta)
        stored += 1
    return {"repo": repo_name, "commits": len(commits),
            "stored": stored, "skipped": skipped, "fallback": fallback}


async def recall_coding(mem, query: str, *, repo: Optional[str] = None,
                        k: int = 5) -> list[dict]:
    """Recall coding decision records, optionally within one repo."""
    results, _stats = await mem.recall(query, k=max(k * 4, 20),
                                       tier="episodic")
    out: list[dict] = []
    for r in results:
        meta = r.metadata or {}
        if meta.get("layer") != "coding":
            continue
        if repo and meta.get("repo") != repo:
            continue
        out.append({"text": r.text, "repo": meta.get("repo"),
                    "commit": meta.get("commit"),
                    "files": meta.get("files"),
                    "chain_prev": meta.get("chain_prev")})
        if len(out) >= k:
            break
    return out
