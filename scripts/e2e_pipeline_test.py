#!/usr/bin/env python3
"""
e2e_pipeline_test.py — full end-to-end test of the Hy-Memory pipeline.

    chunk -> D1 gate -> Hy-Memory extract -> update ops -> temp bank
    (real MemoryStore + local 384-dim embeddings) -> recall queries

Everything writes to a THROWAWAY Chroma dir under /tmp; the real bank
is never touched. Hy-Memory is loaded for the test and unloaded after.
Exit code 0 only if every hard stage passes.
"""
from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["DUCKBOT_EMBEDDING"] = "local"  # same pin as run-watcher.sh

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL = "hy-memory-30b-a3b-mlx"  # MLX 4-bit (GGUF retired 2026-10-16, Duckets)
LMS = Path.home() / ".lmstudio" / "bin" / "lms"

TEST_DOC = """# E2E test note (synthetic)

## Morning chat
Duckets mentioned over coffee that he has switched from drip coffee to a
double espresso every morning, and he wants the espresso preference
remembered as his current one for grocery orders and morning briefings.
He also said Jessica's church bake sale is next Saturday and he is
bringing two trays of brownies for it, which belongs on the family
calendar so nobody double-books that morning by accident or otherwise.

## Noise
lol ok yeah sounds good. asdf placeholder sample text nothing here at
all worth keeping. umm hmm whatever, talk later maybe, idk. test test
test one two three. filler filler filler words words words going nowhere
in particular, just a paragraph of pure conversational static and noise.

## Mining update
Duckets restarted the SRBMiner on Batman-3 tonight and it is mining
Pearl again. He said the gaming break is over and the miner should be
treated as running from now on, unless he says otherwise in a future
update. This supersedes the earlier note that the miner was staying off.

## RAG stack decision
For the DuckBot RAG stack the decision stands that embeddings stay on
the local 384 dimensional model into Chroma, and the dimension source
must never be switched, because changing it would invalidate the whole
existing bank. The reranker remains the Qwen reranker and the decay
pruning stays FSRS based, exactly as previously decided by Duckets.
"""

SEEDED_BANK = [
    "Duckets drinks drip coffee every morning.",
    "Duckets stopped mining on Batman-3 on Oct 16 to game; the miner stays off until he restarts it himself.",
    "Duckets' favorite color is green.",
]

RESULTS: list[tuple[str, bool, str]] = []


def stage(name: str, ok: bool, detail: str) -> bool:
    RESULTS.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    return ok


def main() -> int:
    from src.chunk import Chunk, chunk_markdown
    from src.gate import gate_chunk
    from src.hy_memory import extract_memories_hy, plan_memory_updates

    # -- S1: chunk ---------------------------------------------------------
    chunks = [c for c in chunk_markdown(TEST_DOC) if len((c.text or "").strip()) >= 120]
    stage("S1 chunk", len(chunks) >= 3, f"{len(chunks)} chunks from the test doc")
    if len(chunks) < 3:
        return _finish()

    # -- S2: gate ----------------------------------------------------------
    kept, skipped = [], 0
    for ch in chunks:
        d = gate_chunk(ch.text)
        if d.gated and d.verdict == "skip":
            skipped += 1
        else:
            kept.append((ch, d))
    noise_skipped = any(
        "asdf placeholder" in (c.text or "") for c in chunks
    ) and skipped >= 1
    stage("S2 gate", skipped >= 1, f"{skipped} skipped, {len(kept)} stored "
          f"(noise chunk skipped: {noise_skipped})")

    # -- load writer -------------------------------------------------------
    subprocess.run([str(LMS), "load", MODEL, "--context-length", "8192", "-y"],
                   capture_output=True, timeout=300)
    try:
        # -- S3: extract ---------------------------------------------------
        facts: list[str] = []
        for ch, _d in kept:
            facts.extend(extract_memories_hy(ch.text, model=MODEL))
        blob = " || ".join(facts).lower()
        checks = {
            "espresso": "espresso" in blob,
            "mining/Pearl": ("pearl" in blob or "mining" in blob),
            "384-dim decision": "384" in blob,
            "bake sale": "bake sale" in blob or "brownies" in blob,
        }
        stage("S3 extract", all(checks.values()),
              f"{len(facts)} facts; topics found: "
              + ", ".join(k for k, v in checks.items() if v)
              + ("; MISSING: " + ", ".join(k for k, v in checks.items() if not v)
                 if not all(checks.values()) else ""))

        # -- S4: update ops -------------------------------------------------
        new_text = "\n\n".join(facts)
        ops = plan_memory_updates(new_text, SEEDED_BANK, model=MODEL)
        coffee_hit = any("coffee" in (o.get("target") or "") for o in ops)
        mining_hit = any("mining" in (o.get("target") or "") or
                         "miner" in (o.get("target") or "") for o in ops)
        green_hit = any("green" in (o.get("target") or "") for o in ops)
        stage("S4 update ops", coffee_hit and mining_hit,
              f"{len(ops)} ops; coffee targeted: {coffee_hit}, mining targeted: "
              f"{mining_hit}, untouched control (green) targeted: {green_hit}")
        for o in ops:
            print(f"       op: {o}")

        # apply: adds + updates land; retires are review-only (Phase 5 rule)
        bank = list(SEEDED_BANK)
        for o in ops:
            if o["op"] == "add" and o.get("text"):
                bank.append(o["text"])
            elif o["op"] == "update" and o.get("target") and o.get("text"):
                bank = [o["text"] if b == o["target"] else b for b in bank]
            elif o["op"] == "retire":
                print(f"       retire HELD for review: {o.get('target')}")
        bank.extend(f for f in facts if f not in bank)
        # superseded = the seeded assertion is gone (the replacement text
        # legitimately *mentions* drip coffee as what he switched from)
        seeded_gone = "Duckets drinks drip coffee every morning." not in bank
        stage("S4b apply", any("espresso" in b.lower() for b in bank) and seeded_gone,
              f"bank now holds {len(bank)} entries; seeded drip-coffee "
              f"assertion superseded: {seeded_gone}")
    finally:
        subprocess.run([str(LMS), "unload", MODEL], capture_output=True, timeout=120)

    # -- S5: store + recall in a throwaway bank -----------------------------
    tmp = Path(tempfile.mkdtemp(prefix="duckbot-e2e-"))
    try:
        ok = asyncio.run(_store_and_recall(bank, tmp, Chunk))
        stage("S5 store+recall", ok, "temp Chroma bank answered all 3 recall probes"
              if ok else "a recall probe missed its fact")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return _finish()


async def _store_and_recall(bank: list[str], tmp: Path, Chunk) -> bool:
    from src.embeddings import get_default_provider
    from src.store import MemoryStore
    from src.tier import Tier

    provider = get_default_provider()
    store = MemoryStore(persist_dir=tmp, embedding_dim=384,
                        embedding_provider_name="local")
    chunks = [Chunk(text=t, source_path="e2e-test", start_char=0,
                    end_char=len(t), chunk_index=i, total_chunks=len(bank))
              for i, t in enumerate(bank)]
    embs = await provider.embed(bank)
    added = await store.add_chunks(chunks, embs, Tier.SEMANTIC)
    if added < len(bank):
        print(f"       only {added}/{len(bank)} entries stored")
        return False
    probes = {
        "what coffee does Duckets drink now": "espresso",
        "is the miner on Batman-3 running": "pearl",
        "which embedding dimension does the RAG stack use": "384",
    }
    all_ok = True
    for q, needle in probes.items():
        qe = await provider.embed_one(q)
        hits = store.query(qe, tier=Tier.SEMANTIC, n_results=2)
        found = any(needle in h["text"].lower() for h in hits)
        print(f"       probe '{q}' -> {'HIT' if found else 'MISS'} "
              f"(top: {hits[0]['text'][:70] if hits else 'none'}...)")
        all_ok = all_ok and found
    return all_ok


def _finish() -> int:
    hard_fail = [n for n, ok, _ in RESULTS if not ok]
    print("=" * 60)
    print(f"E2E RESULT: {'ALL STAGES PASS' if not hard_fail else 'FAILED: ' + ', '.join(hard_fail)}")
    return 0 if not hard_fail else 1


if __name__ == "__main__":
    raise SystemExit(main())
