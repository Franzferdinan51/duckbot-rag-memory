#!/usr/bin/env python3
"""
Rebuild a corrupted ChromaDB HNSW index from SQLite + FTS metadata.

When a ChromaDB Rust hnswlib binding segfaults during access (typically
after a partial write, interrupted ingest, or chroma version mismatch),
the only fix is to drop the broken HNSW file and re-create it from the
text + metadata that's preserved in chroma.sqlite3.

This script:
  1. Reads all embeddings for a target collection from chroma.sqlite3
     (joined with embedding_fulltext_search_content for the text)
  2. Batches the texts through LM Studio's /v1/embeddings endpoint
     (default model: text-embedding-nomic-embed-text-v1.5, 768-dim)
  3. Re-adds them to the ChromaDB collection using the SAME embedding_ids
     so existing FTS metadata stays in sync

Non-destructive: the original HNSW file should be moved aside (NOT deleted)
before running this script. This script writes a fresh HNSW index.

Usage:
    .venv/bin/python scripts/rebuild-corrupt-hnsw.py <collection_name> \
        [--batch-size 32] [--model text-embedding-nomic-embed-text-v1.5] \
        [--lmstudio-url http://127.0.0.1:1234/v1] [--limit N]

Environment:
    LMSTUDIO_API_KEY  — bearer token (loaded from .env if present)
    LMSTUDIO_URL      — endpoint (default http://127.0.0.1:1234/v1)
"""
import argparse
import asyncio
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

# Load .env
ENV_PATH = Path(__file__).parent.parent / ".env"
if ENV_PATH.exists():
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() not in os.environ:
            os.environ[k.strip()] = v.strip()

import httpx  # noqa: E402

CHROMA_PATH = Path(__file__).parent.parent / "data" / "chroma" / "chroma.sqlite3"


def read_entries(conn, collection_name: str, limit: int | None = None):
    """Read all (id, embedding_id, text, metadata) for a collection."""
    cur = conn.execute(
        "SELECT id, name FROM collections WHERE name = ?", (collection_name,)
    )
    row = cur.fetchone()
    if not row:
        raise SystemExit(f"Collection {collection_name!r} not found in chroma.sqlite3")
    collection_id = row[0]

    # Find the METADATA segment for this collection — that's where the
    # text and per-chunk metadata live. The VECTOR segment holds the HNSW
    # index; if it's been wiped (corruption recovery), embeddings get
    # re-added there by chroma when we call coll.add() with embeddings.
    cur = conn.execute(
        "SELECT id FROM segments WHERE collection = ? AND scope = 'METADATA'",
        (collection_id,),
    )
    row = cur.fetchone()
    if not row:
        raise SystemExit(f"No METADATA segment for collection {collection_name!r}")
    seg_id = row[0]

    sql = """
        SELECT e.id, e.embedding_id, fts.c0 AS text
        FROM embeddings e
        JOIN embedding_fulltext_search_content fts ON fts.id = e.id
        WHERE e.segment_id = ?
        ORDER BY e.id
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    cur = conn.execute(sql, (seg_id,))
    raw_entries = cur.fetchall()

    entries = []
    for eid, embedding_id, text in raw_entries:
        # Reconstruct metadata from embedding_metadata
        meta_cur = conn.execute(
            "SELECT key, string_value, int_value, float_value, bool_value "
            "FROM embedding_metadata WHERE id = ?",
            (eid,),
        )
        meta = {}
        for key, sv, iv, fv, bv in meta_cur.fetchall():
            if sv is not None:
                meta[key] = sv
            elif iv is not None:
                meta[key] = iv
            elif fv is not None:
                meta[key] = fv
            elif bv is not None:
                meta[key] = bool(bv)
        # Strip chroma-internal keys that conflict with add() shape
        meta.pop("chroma:document", None)
        entries.append((eid, embedding_id, text, meta))
    return entries


def batched(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


async def embed_batch(client: httpx.AsyncClient, url: str, model: str, texts: list[str]):
    """Call LM Studio /v1/embeddings for a batch of texts."""
    r = await client.post(
        f"{url}/embeddings",
        headers={
            "Authorization": f"Bearer {os.environ.get('LMSTUDIO_API_KEY', 'lm-studio')}",
            "Content-Type": "application/json",
        },
        json={"input": texts, "model": model},
        timeout=120.0,
    )
    r.raise_for_status()
    data = r.json()
    # Sort by index to preserve order
    return [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("collection", help="ChromaDB collection name to rebuild")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--model", default=os.environ.get("LMSTUDIO_EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5"))
    ap.add_argument("--lmstudio-url", default=os.environ.get("LMSTUDIO_URL", "http://127.0.0.1:1234/v1"))
    ap.add_argument("--limit", type=int, default=None, help="Rebuild only first N (for testing)")
    ap.add_argument("--dry-run", action="store_true", help="Read entries, don't write")
    args = ap.parse_args()

    print(f"Reading from {CHROMA_PATH}", flush=True)
    conn = sqlite3.connect(str(CHROMA_PATH))
    conn.row_factory = sqlite3.Row
    entries = read_entries(conn, args.collection, limit=args.limit)
    conn.close()
    print(f"Found {len(entries)} entries in collection {args.collection!r}", flush=True)
    if not entries:
        return
    if args.limit:
        print(f"  (limited to first {args.limit})", flush=True)

    if args.dry_run:
        print("Dry run — sample entry:", flush=True)
        eid, embedding_id, text, meta = entries[0]
        print(f"  id={eid}, embedding_id={embedding_id}", flush=True)
        print(f"  text[0:120] = {text[:120]!r}", flush=True)
        print(f"  metadata keys = {list(meta.keys())}", flush=True)
        return

    # Late import chromadb so --dry-run doesn't need it
    import chromadb  # noqa: E402
    from chromadb.config import Settings  # noqa: E402

    print(f"Opening chromadb at {CHROMA_PATH.parent}", flush=True)
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH.parent), settings=Settings(anonymized_telemetry=False)
    )
    coll = client.get_collection(args.collection)
    print(f"Re-adding {len(entries)} entries via LM Studio ({args.model})...", flush=True)

    t0 = time.time()
    total_added = 0
    failed = 0
    async with httpx.AsyncClient() as http:
        for batch in batched(entries, args.batch_size):
            embedding_ids = [e[1] for e in batch]
            texts = [e[2] for e in batch]
            metas = [e[3] for e in batch]

            try:
                embeddings = await embed_batch(http, args.lmstudio_url, args.model, texts)
            except Exception as e:
                failed += len(batch)
                print(f"  BATCH FAILED ({len(batch)} items): {type(e).__name__}: {e}", flush=True)
                continue

            try:
                coll.add(
                    ids=embedding_ids,
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metas,
                )
                total_added += len(batch)
            except Exception as e:
                failed += len(batch)
                print(f"  ADD FAILED ({len(batch)} items): {type(e).__name__}: {e}", flush=True)
                continue

            elapsed = time.time() - t0
            rate = total_added / elapsed if elapsed > 0 else 0
            print(
                f"  {total_added}/{len(entries)} added ({rate:.1f}/s, {failed} failed)",
                flush=True,
            )

    elapsed = time.time() - t0
    print(f"\nDone: {total_added} added, {failed} failed, {elapsed:.1f}s", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())