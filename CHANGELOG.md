# Changelog

## 0.6.0 — 2026-06-23 — Pluggable backend seam (L14)

The brain can now swap vector stores without touching callers. Existing
code (`MemoryStore`, query pipeline, MCP server) keeps its current API;
internally it delegates to a `VectorBackend` selected by `DUCKBOT_BACKEND`.

Pattern source: `MemPalace/mempalace` `backends/base.py` (MIT).

### L14 — Pluggable backend seam

- **`src/backends/base.py`** — `VectorBackend` ABC + `VectorHit` /
  `BackendStats` / `TierStats` dataclasses. Five required methods:
  `add_chunks`, `query`, `bm25_query`, `delete`, `stats`. Plus
  `register_backend(name, "pkg.mod.Class")` for runtime plugins.
- **`src/backends/chroma.py`** — `ChromaBackend` wrapping the existing
  ChromaDB code. One collection per tier, 8 KB verbatim cap, lazy load.
- **`src/backends/qdrant.py`** — `QdrantBackend` stub (Apache-2.0).
  Raises helpful `ImportError` on missing deps, `NotImplementedError`
  on unimplemented methods.
- **`src/backends/lancedb.py`** — `LanceDBBackend` stub (Apache-2.0).
  Same shape as the Qdrant stub.
- **`src/backends/__init__.py`** — `get_backend(name=None, **kwargs)`
  resolves by name or `DUCKBOT_BACKEND` env var. `list_backends()`
  returns built-in + runtime-registered backends.
- **`src/store.py`** — refactored to delegate to the configured backend.
  All legacy methods preserved (`add_chunks`, `query`, `bm25_query`,
  `stats`, `mark_ingested`, `mark_queried`, `reset`, `collection_for`).
  Existing tests/callers untouched.

### Verification

- 342/342 tests pass (was 306; +36 from L14).
- End-to-end: `Brain.recall()` still works through the new backend.
- OpenClaw stdio adapter still works end-to-end through the new backend.
- Pattern source verified via GitHub API: MemPalace 56k stars, MIT.

## 0.5.0 — 2026-06-23 — Cross-runtime integration (L16)

Duckets pointed us at OpenClaw (`openclaw/openclaw`, 380k stars) and Hermes
(`NousResearch/hermes-agent`, 201k stars). Both have native memory plugin
systems. We now ship a plugin for each.

### L16 — Hermes MemoryProvider plugin

- **`src/plugins/memory/duckbot_brain/`** — Hermes plugin implementing the
  `MemoryProvider` ABC from `agent/memory_provider.py`.
  - `register(ctx)` — standard plugin entry; pushes the provider into the
    Hermes plugin context.
  - `initialize(session_id, **kwargs)` — per ABC; honors `agent_context`
    (skip writes for `cron`/`subagent`/`flush` contexts).
  - `prefetch(query)` — fast recall (k=3, no rerank/decay) for prompt
    injection before each turn. Returns formatted `[memory]` block.
  - `sync_turn(user, assistant)` — non-blocking background write to the
    brain via `ThreadPoolExecutor`. Skip-on-non-primary honored.
  - `system_prompt_block()` — static text describing the brain tools.
  - `get_tool_schemas()` — three OpenAI-function-call schemas: brain_recall,
    brain_recall_verbatim, brain_reflect.
  - `handle_tool_call(name, args)` — dispatches tool calls. brain_recall
    and brain_recall_verbatim delegate to `Brain.recall()` /
