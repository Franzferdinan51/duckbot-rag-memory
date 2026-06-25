# Changelog

## 0.12.0 — 2026-06-24 — MemPalace + mem0 inspired upgrades

Borrowing the highest-value open-source patterns from MemPalace
(verbatim-first, 96.6% R@5 on LongMemEval) and mem0 (production-tested
memory layer) — all local-first, no new paid deps.

### Added

- **`brain_wake_up` MCP tool** — one-call session-start context load.
  Returns top-k recent memories (filtered to drop superseded chunks),
  active memory blocks, graph summary, FSRS review queue, and stats.
  MemPalace-inspired design: agents on session start do ONE call instead
  of N round-trips. Wired as the canonical pre-flight for Hermes +
  OpenClaw session hooks.

- **`scripts/hermes-preflight.sh` + `scripts/hermes-postflight.sh`** —
  Hermes session-start/-end hook scripts that call `wake-up` and
  `reflect` respectively. Drop them into `~/.hermesrc` or as
  `SessionStart`/`SessionEnd` hooks in any MCP-compatible agent.

- **MemPalace-style hybrid v4 retrieval boosts** in `query.py`:
  - **Keyword boost**: chunks whose text contains the query's exact
    terms (not just BM25-ranked) get a small flat bonus (0.005/hit,
    capped at 0.03). Improves precision for proper nouns and IDs.
  - **Temporal-proximity boost**: recently-ingested memories score
    higher (half-life of 30 days, +0.02 today, asymptotes to 0).
    Improves "what did we just decide" recall.
  Both are opt-in via `DUCKBOT_KEYWORD_BOOST` / `DUCKBOT_TEMPORAL_BOOST`
  env vars (default ON). Zero new deps, zero API cost.

- **mem0-style conflict detection** in `Memory.remember()` — when a new
  chunk is within cosine distance 0.08 of an existing one in the same
  tier, the old chunk is marked `superseded_by` (with timestamp) and the
  new chunk gets a `supersedes` backref. Preserves audit trail; old
  recall results still resolve but new queries prefer the fresh fact.
  `brain_wake_up` automatically drops superseded chunks from its output.

- **`CLI wake-up` and `CLI reflect` subcommands** — `python -m src.cli
  wake-up` prints a markdown context block ready to paste into any
  agent's context window; `python -m src.cli reflect --days 7` runs
  consolidation.

- **OpenClaw skill manifest** at `skills/openclaw-imports/duckbot-rag-memory/SKILL.md`
  — describes the full tool surface for the OpenClaw agent platform.

### Tests

- 11 new regression tests in `tests/test_bugfixes_v0_11_3.py` for the
  keyword/temporal boost phases, conflict detection, brain_wake_up
  method, MCP tool registration, CLI subcommands, and hook scripts.
- Full suite: **573 passing** (was 564 in v0.11.12).

## 0.11.12 — 2026-06-24 — Make it a real brain

Three wire-ups that turn the storage layer into an actual learning memory.

### Added — memories strengthen when you use them

- **`Memory.recall()` now bumps `fsrs_stability_days` on every returned
  chunk** via `decay.bump_stability()`. Previously, `recall_count` and
  `last_recalled_at` were updated but the FSRS forgetting curve stayed
  constant — so the brain never actually learned from usage patterns.
  After this change, a chunk that's been recalled 10x is materially
  harder to forget than a fresh one. This is THE core spaced-repetition
  loop.

### Added — dreaming actually distills

- **`DreamingBridge.cycle()` now ingests a distilled semantic-tier chunk**
  in addition to writing the dream file. Previously, `cycle()` produced a
  bullet-list of previews and called that "distillation"; future recalls
  never saw the summary. Now a single `Dream distillation YYYY-MM-DD`
  chunk lands in the semantic tier, so the brain can surface the
  distilled signal directly. The dream file is still written for
  OpenClaw's own dreamer to layer on top.

- **`DreamCycleResult` gained `distilled_into_semantic: bool`** so
  callers can detect distillation failures (non-fatal: dream file is
  still written even if the semantic remember() fails).

### Added — block_rethink is no longer a no-op

- **`Brain.block_rethink(name, instruction)` now appends the instruction
  to a JSONL queue** at `data/blocks/<name>.rethink.jsonl`. An external
  LLM-driven script (or the dashboard) drains the queue:

  1. read the block via `block_read` (returns `queued_instructions`)
  2. run each queued instruction through the LLM
  3. write the result via `block_write`
  4. clear the queue file

  This makes `block_rethink` a real durable signal the user can act on
  later, not a silent no-op. The response includes `queue_len` and
  `queue_path` so scripts can monitor the queue without parsing
  internals.

- **`Brain.block_read(name)` now includes `queued_instructions`** in its
  output dict so callers see pending rethink entries automatically.

### Tests

- 4 new regression tests in `tests/test_bugfixes_v0_11_3.py` for the
  bump-on-recall, dream distillation, queue-write, and queue-read paths.
- Full suite: **564 passing** (was 560 in v0.11.11).

## 0.11.5 — 2026-06-24 — Audit-driven bug fixes

Correctness bugs surfaced by a focused audit of the connectors, watcher, CLI,
and backends. Each fix has a regression test in
`tests/test_bugfixes_v0_11_3.py`.

### Fixed

- **`connectors/active_memory.py`** — `memory_query`, `memory_recent`, and
  `memory_store` assumed `Brain.recall()` / `Brain.remember()` returned objects
  with a `.results` / `.chunk_id` shape. In reality `recall()` returns a plain
  `list[RecallResult]` and `remember()` returns a `RememberResult` dataclass.
  Every active-memory call was raising `AttributeError`. Now iterates the list
  directly and unwraps `RememberResult.chunk_id`.
- **`connectors/dreaming.py`** — `Memory.recall()` returns a
  `tuple[list[QueryResult], QueryStats]`; the dreaming bridge called `r.results`
  and crashed on both episodic and procedural recall. Now unpacks the tuple.
  The previously-swallowed procedural-recall exception is also logged.
- **`connectors/hermes.py`, `connectors/openclaw.py`** — `reflect()` used
  raw `asyncio.run()` from a sync helper. When called from inside a running
  loop (the MCP server, FastMCP, asyncio pytest) it raised `RuntimeError`.
  Now routes through the shared `_run_async` bridge from `connectors/base.py`.
- **`watcher.py`** — the watchdog-not-installed fallback returned
  `PollingHandler(...).run()`, an un-awaited coroutine — the fallback silently
  did nothing. Now wrapped in `asyncio.run(...)` to match the polling branch
  above it.
- **`cli.py`** (`cmd_compact`) — `VACUUM` was executed inside an implicit
  SQLite transaction (`with sqlite3.connect(...)`), which raises
  `OperationalError: cannot VACUUM from within a transaction` and was masked
  by the surrounding `except`. Now sets `isolation_level = None` (autocommit)
  before `VACUUM`.
- **`memory.py` (`forget`)** — returned `True` even when nothing was deleted,
  because Chroma's `delete()` is a no-op for unknown ids. Now checks the
  collection for the id first and returns `False` when it wasn't present.
- **`eval.py` (`_is_hit`)** — tier-only eval entries read
  `result_meta.get("tier")`, but `tier` is a top-level `QueryResult`
  attribute and was never written to metadata — so every tier-only eval
  entry silently scored 0. `_is_hit` now takes an explicit `result_tier`
  parameter (with a metadata fallback), and the eval loop threads `r.tier`
  through.
- **`backends/chroma.py`** — query-across-all-tiers used
  `n_results // len(tiers)`, which floors and under-fetches (5 requested
  across 4 tiers returned 4 candidates). Now uses `math.ceil` so the union
  always covers at least `n_results`.
- **`consolidate.py`** — the `user-said` FACT_PATTERN used the regex
  `(?:he/she)`, which matches the literal string "he/she" (slash and all),
  not "he or she". Now `(?:he|she)`.
- **`injection_scan.py`** — the `zero_width_chars` pattern range was too
  broad and flagged legitimate formatting: line/paragraph separators
  (`\u2028`/`\u2029`), narrow no-break space (`\u202F`), medium math space
  (`\u205F`), word joiner (`\u2060`), and invisible math operators
  (`\u2061`-`\u2064`). Tightened to the actual injection/spoofing vectors:
  zero-width chars (`\u200B`-`\u200F`), directional embedding/override
  (`\u202A`-`\u202E`), directional isolates + deprecated formatting
  (`\u2066`-`\u206F`), and BOM (`\uFEFF`). Reduces false positives without
  weakening detection of real RTL-override / directional-isolate spoofs.

### Tests

- Added `tests/test_bugfixes_v0_11_3.py` (12 regression tests).
- Updated `tests/test_v0_11_integration.py` `FakeBrain` / `FakeMemory`
  stubs to match the real `recall()` / `remember()` return contracts — the
  old stubs simulated the *buggy* shape and were passing against the bug.
- Full suite: **529 passing** (was 517).

## 0.11.4 — 2026-06-24 — Reliability pass

### Fixed

- Dashboard tests no longer expire with wall-clock time: reports and the
  24-hour summary accept an optional clock for deterministic validation while
  production still uses the current time by default.
- `build_report(chroma_path=...)` now reads the specified Chroma directory
  rather than silently querying the default store.
- `scripts/duckbot-ask` and `scripts/brain-recall.sh` are executable as
  documented; the wrapper help now correctly reports its 500-character
  default preview limit.
- Watcher docstrings now state the actual five-minute polling default.

## 0.11.2 — 2026-06-24 — LM Studio spam hotfix + enhanced brain

LM Studio's embed endpoint was being hammered by the duckbot-rag-memory
stack. This release fixes three root causes and adds the enhanced brain
— a system that actively writes memories back to agent context files
so agents don't start each session from a blank slate.

**Root causes fixed:**
1. **No embed-result cache.** Every `brain_decay_status`, `brain_fsrs_review`,
   and watcher poll re-embedded the same chunks.
2. **Each call opened a fresh `httpx.AsyncClient`.** With v0.10/v0.11's
   three concurrent embed paths (Layer 6 OpenClaw connector, Layer 16
   Hermes plugin, MCP server), bursts collided at LM Studio's single-threaded
   HTTP server and triggered `ERR_HTTP_HEADERS_SENT`.
3. **No rate limiter.** All callers slammed LM Studio's `/v1/embeddings`
   endpoint with no global throttle.
4. **Per-call dim probe.** `Memory()` ran `embed_one("dim probe")` on every
   instantiation in long-lived daemons. Now cached at process level.

### Added — src/embeddings.py
- **Shared `httpx.AsyncClient` singleton.** One connection pool per process.
  Lazy lock ensures correct loop attachment in pytest.
- **LRU result cache (`_EmbedCache`).** Default 4096 entries. Keyed on
  `(sha256(text), model_name)`. Set `DUCKBOT_EMBED_CACHE_SIZE=0` to disable.
- **Async token-bucket rate limiter (`_TokenBucket`).** Default 60 req/min.
  Set `DUCKBOT_EMBED_RPM=N` to override.
- **Process-level dim-probe cache.** `LMStudioEmbeddings._resolve_dim()` and
  `Memory._ensure_initialized()` both short-circuit on cache hit. Failed probes
  are cached as `None` sentinel — no repeated retries on broken endpoints.

### Added — src/watcher.py
- **Content-hash dedup on file sync.** No-op file rewrites skip re-ingest entirely.

### Added — src/mcp_server.py
- **`brain_inflate` MCP tool.** Recall relevant memories and format them as a
  markdown block ready for agent context. Use when starting a new session,
  task, or when asked "what do I know about X?"
- **`brain_sync` MCP tool.** Write memories back to agent context files:
  - OpenClaw: `~/.openclaw/workspace/memory/{MEMORY,USER,SOUL}.md`
  - Hermes: `~/.hermes/memories/{MEMORY,USER}.md` (char-limited format)
  - Both: `~/.hermes/SOUL.md`
- Tool count: **43 tools** (was 39).

### Added — OpenClaw skill
- `skills/duckbot-brain/SKILL.md` — teaches agents to use `brain_inflate`
  and `brain_sync`. Installs to `~/.openclaw/workspace/skills/`.

### Added — CLI
- `python -m src.cli sync --target openclaw|hermes|both` — call `brain_sync`
  from cron without needing the MCP transport.

### Fixed — src/chunk.py
- `char_offset` now tracks position incrementally across markdown sections
  instead of re-running `text.find(body)` per section (which returned the
  first occurrence, wrong for duplicate section bodies).

### Fixed — src/graph.py, src/blocks.py, src/injection_scan.py
- Hardcoded `Path.home() / "Desktop" / "duckbot-rag-memory"` paths replaced
  with `Path(__file__).resolve().parent.parent / "data" / ...` (repo-relative).

### Tests — tests/test_v0_11_2_hotfix.py + tests/test_dim_probe_cache.py
- 21 new tests covering: cache hit/miss/LRU/disable, token-bucket
  burst/exhaust/refill, shared-client singleton + reopen, end-to-end
  `embed()` caching, watcher content-hash dedup, dim-probe first/cached/failed.
- **Full suite: 500 tests passing.**

### Not changed
- No schema migration.
- No breaking config changes.
- Watcher state format: extended (new `content_hash` field) but the
  old format still loads — missing hash falls through to mtime dedup.

## 0.11.4 — 2026-06-24 — Repo setup for community use

Round of "make this a real project that other humans (and agents) can
actually use" changes. No code changes. No new tools. No breaking API.

### Added

- **`.github/workflows/ci.yml`** — GitHub Actions CI on push + PR.
  Matrix: Python 3.11 + 3.12 × ubuntu/macos/windows. Runs secret-scan
  first (fast-fail on accidental key commits), then pytest. LM-Studio
  integration tests are auto-skipped in CI.
- **`.github/ISSUE_TEMPLATE/bug_report.yml`** — severity, OS, embedding
  provider, what-happened, expected, repro, environment, workarounds.
- **`.github/ISSUE_TEMPLATE/feature_request.yml`** — problem, proposal,
  alternatives, cross-platform impact, backward-compat.
- **`.github/ISSUE_TEMPLATE/config.yml`** — links to docs, discussions,
  security disclosure; disables blank issues.
- **`.github/PULL_REQUEST_TEMPLATE.md`** — checklist for: secret-scan,
  no-deletions, cross-platform, tests, CHANGELOG, requirements update.
- **`SECURITY.md`** — supported versions, private disclosure channels,
  48h acknowledgment + 30d CVE SLA, hardening checklist for users,
  high-value code paths to review.
- **`CONTRIBUTING.md`** — project values, dev setup, coding conventions,
  review process, what we won't merge, license (MIT).
- **`tests/__init__.py`** — makes `tests/` a Python package so
  `from tests._mock_embedder import MockEmbeddings` works (was failing
  test collection before). **Fixes 3 pre-existing test collection
  errors**, taking total collected tests from 480 → 512.

### Changed

- **`pytest.ini`** — added `pythonpath = . tests` so the package is
  importable from the repo root without manual `sys.path` hacks.
- **`AGENTS.md`** — quick-start now shows `duckbot-ask` /
  `brain-recall` / `start-watcher` / `hermes mcp add`. File layout
  diagram expanded to cover the new files. "Integration" section now
  covers both OpenClaw (cron + ingest) AND Hermes Agent (MCP server
  with 43 tools). Cross-platform paths throughout.
- **`README.md`** — status badge updated to v0.11.3, CI badge added,
  Quick Start shows the shell-wrapper usage path alongside the python
  CLI, "Why polling" section updated to 5-min default (was 2s).

### Not changed

- No code changes. No new tools. No new dependencies.
- Public API surface is identical.
- `data/`, `.env`, `__pycache__/`, `.venv/` still gitignored.

## 0.11.3 — 2026-06-24 — duckbot-ask + 5-min watcher default

Two small additions to round out the brain's reach into scripts and
cron jobs. No API changes, no breaking changes.

### Added

- **`scripts/duckbot-ask`** — thin bash wrapper around
  `python -m src.cli query`. Gives cron jobs and one-shot shell
  sessions a one-liner to query the brain:
  ```bash
  duckbot-ask "PRL pool wallet workers"
  duckbot-ask -f compact -n 5 "Duckets correction style"
  duckbot-ask -f snippet "BATMAN container restart recipe"
  ```
  Three output formats: `json` (default, full structured), `compact`
  (one block per result, Telegram-friendly), `snippet` (just the
  first result's text). Loads `.env` itself so `LMSTUDIO_API_KEY`
  never leaks through `ps`. Cross-platform venv detection (mirrors
  `duckbot-memory-mcp.sh`).

- **`scripts/_format_snippet.py`** + **`scripts/_format_compact.py`** —
  the python formatters behind `duckbot-ask`'s `-f snippet` /
  `-f compact`. Same shape as the bash wrapper but standalone so
  Python pipelines can pipe `python -m src.cli query` directly
  through them.

- **`tests/test_duckbot_ask.py`** — 12 tests covering formatter
  unit behavior, bash wrapper structure, and live LM Studio
  integration (5 of the 12 are real brain round-trips; skipped if
  LM Studio is unreachable).

### Changed

- **Default watcher polling interval: 2s → 300s (5 min).** 5 paths
  polled every 2 seconds was 150 polls/minute for no benefit —
  markdown files don't change that often. New default keeps the
  brain fresh within ~5 min without burning cycles. Override with
  `--interval N` on `watcher run`/`watcher daemon`, or in
  `start-watcher.ps1` / `start-watcher.sh`.

## 0.10.1 — 2026-06-23 — Cross-platform MCP stdio fix + README paths

A small follow-up to v0.10.0, prompted by Windows + Hermes-Agent
integrations. No API changes. No breaking changes. Indexing, schema,
and tool definitions are unchanged from v0.10.0.

### Fixes

- **MCP stdio server now self-configures line-buffered I/O at startup.**
  `src/mcp_server.py` calls `sys.stdin.reconfigure(line_buffering=True)` /
  `sys.stdout.reconfigure(line_buffering=True)` (and stderr) before reading
  the first request. Without this, **Windows** block-buffers the subprocess
  stdout in 4-8 KiB chunks and short `initialize` responses (~167 bytes)
  sit in the kernel pipe buffer until the MCP client times out with
  "Connection closed." On macOS and Linux this is a harmless no-op (those
  platforms already flush per-write for line-buffered TTYs). No need for
  `-u` or `PYTHONUNBUFFERED=1` in the launcher anymore — though both still
  work if you have them in your config.

- **README cross-platform guidance.** Added a "Wire it into Hermes Agent"
  block with the exact `hermes mcp add` invocation for macOS/Linux/Windows
  (PowerShell + git-bash variants), and a "Cross-platform paths" table
  mapping the POSIX paths that pepper the rest of the README to their
  Windows equivalents. Includes the `hermes mcp add --args nargs=REMAINDER`
  gotcha — put `--env` flags BEFORE `--args` or they'll get swept into
  the arg list.

### Not changed

- No tool added/removed (still 35).
- No schema migration.
- No `.env` keys added/removed.
- No breaking config changes.

## 0.10.0 — 2026-06-23 — Useful MCP tools extension

Duckets asked: "Also add more MCP tools that are useful." This release
adds 4 new tools that wrap the L8/L9/L13 brain layers and a 5th that
adds forget-by-query semantics. Also fixes 2 latent bugs the new tests
caught.

### New MCP tools (exposed in 3 places)

| Tool | Layer | What it does |
|---|---|---|
| `brain_fsrs_review` / `fsrs_review` | L9 | List chunks due for FSRS-6 spaced-repetition review (R(t,S) < 0.9). Sorted by urgency. Public-domain math. |
| `brain_decay_status` / `decay_status` | L8 | Ebbinghaus decay status (R = e^(-t/S)) for recent chunks, grouped by tier. Shows what's fading. Public-domain math. |
| `brain_forget_by_query` / `forget_by_query` | L14 | Delete the top-k chunks matching a query. Different from `brain_forget(chunk_id=...)` which deletes one chunk. |
| `brain_search_verbatim` / `search_verbatim` | L13 | Exact substring match against verbatim (pre-overlap) source text. Useful when you remember a phrase verbatim. |
| `brain_recall_verbatim` / `recall_verbatim` | L13 | (Already existed in `connectors/openclaw.py`; now also exposed in `mcp_server.py` so standalone MCP clients get it.) |

**Total MCP tool count:** 35 (was 30; +5).
**Total `Brain` method count:** 28 (was 24; +4).

### Bugs caught by the new tests

1. **`brain_stats` in OpenClaw extension referenced nonexistent fields.**
   `adapter.py:_call_tool("brain_stats")` was reading
   `s.chunks_per_tier` and `s.last_query_at` — neither field exists on
   `BrainStats`. The tool would have raised `AttributeError` on first
   call. Fixed to read the real fields (`vector_chunks`,
   `vector_by_tier`, `graph_entities`, `blocks`, `quarantine_*`,
   `generated_at`).

2. **`Brain.search_verbatim` accessed `r.source_path` instead of
   metadata.** `QueryResult` doesn't have a `.source_path` attribute;
   that data lives in `r.metadata["source_path"]`. The new test
   `test_search_verbatim_finds_known_string` caught this on first run.

3. **`asyncio.run()` called from inside a running event loop.**
   `Brain.fsrs_review_queue`, `Brain.decay_status`,
   `Brain.forget_by_query`, `Brain.search_verbatim`, plus the
   pre-existing `Brain.recall`, `Brain.remember`, `Brain.recall_verbatim`,
   all called `asyncio.run(coro)` unconditionally. That works from
   sync code (CLI) but raises
   `RuntimeError: asyncio.run() cannot be called from a running event loop`
   when invoked from inside an MCP server (which already runs an event
   loop). **Fixed with a new `_run_async(coro)` helper** that detects
   whether we're in a running loop and, if so, runs the coroutine in
   a worker thread. Now both sync callers (CLI) and async callers
   (MCP server) work without changes to their code.

### Verification

- **459/459 tests pass** (was 446; +13 from new MCP tools extension tests).
- End-to-end smoke test against the live MCP server:
  - `decay_status` returns 10 chunks sampled, avg retention 0.981, breakdown by tier.
  - `search_verbatim("Stop using local models entirely")` finds the exact
    phrase in `AGENTS.md` with highlight context.
  - `fsrs_review` returns empty queue (no chunks have FSRS state yet —
    L9 is opt-in, never enabled by default).
  - `stats` still works (regression test for the brain_stats fix).

### Files changed

- `src/connectors/base.py` — 4 new methods on `Brain` (`fsrs_review_queue`,
  `decay_status`, `forget_by_query`, `search_verbatim`); new
  `_run_async(coro)` helper; 6 existing methods switched to use it.
- `src/connectors/openclaw.py` — 4 new tools in `TOOL_DEFINITIONS` +
  dispatchers in `handle()`.
- `src/extensions/duckbot_brain/adapter.py` — 4 new tools in
  `_tool_schemas()` + dispatchers; `brain_stats` tool reads the real
  `BrainStats` fields now.
- `src/mcp_server.py` — 5 new tools in `TOOLS` (incl. `recall_verbatim`)
  + handlers in `HANDLERS`.
- `tests/test_mcp_tools_extension.py` — 13 new tests covering all of
  the above.
- `tests/test_openclaw_extension.py` — updated the `test_call_tool_brain_stats_delegates`
  test to assert the real `BrainStats` fields.

## 0.9.1 — 2026-06-23 — Bug fixes from cross-platform audit

After the v0.9.0 push, Duckets asked: "Make sure fix any bugs and push
to main and also update the README." This is the honest audit pass.

### Bugs found and fixed (real, demonstrable)

1. **Windows daemon silently dropped all output** (`src/watcher.py`).
   `_daemon_windows` used `subprocess.DEVNULL` for the detached child's
   stdout AND stderr, so any error or log line from the actual watcher
   was silently lost. On Windows you couldn't tell why the watcher
   crashed. Now redirected to `LOG_PATH` (same as POSIX does via
   `dup2`). Regression test: `test_windows_daemon_redirects_logs_to_file`.

2. **README Quick Start had a non-existent CLI subcommand**.
   Said `./.venv/bin/python -m src.cli watch once`. But `src.cli` only
   has subcommands: `ingest`, `query`, `stats`, `eval`, `consolidate`,
   `reset`, `compact`, `doctor`, `hermes`, `dashboard`. The actual
   "one-shot sync" command is `src.watcher once`. Fixed in README.

3. **launchd plist had hardcoded `/Users/duckets/Desktop/...` paths**.
   The plist was committed with Duckets' absolute path baked in. Anyone
   who cloned the repo elsewhere got a broken plist. Now it's a template
   with `__REPO_ROOT__` placeholders, and `install-macos.sh` does
   `sed "s|__REPO_ROOT__|$REPO_ROOT|g"` before copying to
   `~/Library/LaunchAgents/`. Regression test:
   `test_no_hardcoded_absolute_paths_in_scripts`,
   `test_plist_is_a_template`, `test_plist_substitution_round_trip`.

4. **`start-watcher.sh` had a hardcoded absolute path**.
   `cd /Users/duckets/Desktop/duckbot-rag-memory` at the top. Replaced
   with `cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."` to
   derive `REPO_ROOT` from the script's own location. Regression test:
   `test_start_watcher_sh_uses_relative_paths`.

5. **`install.ps1` required `git` on PATH to resolve the repo root**.
   On locked-down Windows boxes without Git for Windows, `git rev-parse
   --show-toplevel` returns null and the script bails. Added a
   `Resolve-RepoRoot` fallback that walks up from `$PSScriptRoot` until
   it finds `src\watcher.py` — an unambiguous marker file. Regression
   test: `test_install_ps1_has_repo_fallback`.

### Honesty about what was missed in v0.9.0

When I pushed v0.9.0, I claimed "it works cross-platform" but I was
running on a Mac. Several of these bugs would have been caught by a
proper `pwsh` syntax check on a Windows box, or by running the
launchd plist on a fresh Mac where the absolute path doesn't exist.
The audit this round was the right thing — but I should have done it
before pushing v0.9.0 in the first place.

### Verification

- 446/446 tests pass (was 439; +7 from regression tests).
- Bash syntax check on all 5 .sh scripts: pass.
- `git status` clean (only the intended files changed).
- Secret-scan clean.
- Pre-commit hook approved the commit.

## 0.9.0 — 2026-06-23 — Full cross-platform support (Win/Mac/Linux)

After the v0.8.0 push, Duckets asked: "make sure the WHOLE thing is
cross platform." That means not just the Chroma enhancements, but the
watcher daemon, the install scripts, the embeddings module, and every
other Python file under `src/`. This release audits + fixes the
remaining cross-platform issues.

### Watcher daemon: real cross-platform `daemon` subcommand

- **`src/watcher.py`** — `cmd_daemon` now dispatches by platform:
  - On POSIX (macOS, Linux): classic double-fork + setsid. Same as
    before, just extracted into `_daemon_posix()`.
  - On Windows: `subprocess.Popen` with `DETACHED_PROCESS` +
    `CREATE_NEW_PROCESS_GROUP` (0x8 | 0x200), which is the Windows
    equivalent of detaching from the controlling terminal. The detached
    child runs `python -m src.watcher run ...` in its own process
    group, surviving parent exit.
- **PID file at `data/watcher.pid`** works identically on all three
  OSes. `watcher status` and `watcher stop` work everywhere.
- **`/dev/null` replaced with `os.devnull`** so the POSIX branch doesn't
  crash if the open path is `/dev/null` on a system that doesn't have
  it (Windows, weird POSIX variants).
- **`os.kill(pid, signal.SIGTERM)`** replaced with a `getattr(signal,
  "SIGTERM", getattr(signal, "SIGBREAK", 15))` lookup. `SIGBREAK` is
  the Windows equivalent; `15` is the universal "kill" fallback.

### `embeddings.py`: `os.path.join` → `pathlib`

- The `.env` file loader now uses `Path(__file__).resolve().parent.parent
  / ".env"` instead of `os.path.join(os.path.dirname(os.path.dirname
  (...)), ".env")`. Same behavior, but the Path form is explicit about
  cross-platform path handling and easier to read.

### New scripts for full cross-platform install

- **`scripts/install.sh`** — generic POSIX bootstrap (venv + deps +
  `.env`). Works on macOS and Linux. No service integration.
- **`scripts/install-macos.sh`** (renamed from `install.sh`) — adds
  launchd plist install. macOS only.
- **`scripts/install-linux.sh`** — new. Writes a systemd user unit to
  `~/.config/systemd/user/duckbot-memory-watcher.service` and runs
  `systemctl --user enable --now`. Works on any distro with systemd
  (Ubuntu 16.04+, Debian 9+, Fedora, Arch, etc.).
- **`scripts/install.ps1`** — new. Windows bootstrap (venv + deps +
  `.env`) + registers a Task Scheduler task that runs the watcher at
  logon, with `RestartCount: 5, RestartInterval: 1 minute`. Visible in
  Task Scheduler UI as `DuckBotMemoryWatcher`.
- **`scripts/start-watcher.ps1`** — new. Cross-platform companion to
  `start-watcher.sh`. Use `pwsh scripts/start-watcher.ps1` to start in
  background, `-Foreground` to run in current console, `-Status` to
  check, `-Stop` to stop, `-Log` to tail logs.
- **`scripts/start-watcher.sh`** — existing POSIX launcher, unchanged.

### Audit + test coverage

- **`tests/test_cross_platform.py`** — 20 new tests covering:
  - Watcher module imports on any platform (no `os.fork` at import).
  - `_daemon_windows` and `_daemon_posix` exist and are called by
    `cmd_daemon` based on `sys.platform`.
  - `/dev/null` is never referenced as a literal in the watcher.
  - `os.chmod` is guarded with `hasattr(os, "chmod")` so Windows
    doesn't crash.
  - `embeddings.py` uses `pathlib`, not `os.path.join`.
  - All 5 OS-specific scripts exist (install.sh, install-macos.sh,
    install-linux.sh, install.ps1, start-watcher.ps1).
  - All bash scripts pass `bash -n` syntax check.
  - **No `os.path.join` anywhere in `src/`** (audit sweep).

### Verification

- 439/439 tests pass (was 419; +20 from new test file).
- Bash syntax check on all 5 .sh scripts: pass.
- PowerShell files are structurally valid (allowlist marker,
  `[CmdletBinding()]`, `param(` block, referenced paths exist).
- `data/watcher.pid` is a portable `Path`, works on Win/Mac/Linux.
- All Python code in `src/` is `pathlib`-based.
- All subprocess calls use `subprocess.DEVNULL`, not `/dev/null`.

### Still on the wishlist (not done in this release)

- **GitHub Actions matrix CI** that runs on `windows-latest` +
  `ubuntu-latest` + `macos-latest` to actually exercise the full
  cross-platform stack end-to-end. Would catch Windows-specific
  issues that this Mac can't reproduce (e.g. `os.fork` AttributeError,
  path-length limits, file-locking races). Recommended next step.
- **Live PS1 syntax check** — `pwsh` is not installed on this Mac, so
  the PowerShell files were structurally validated (markers, blocks,
  referenced paths) but not actually parsed. A Windows runner in CI
  would close that gap.

## 0.8.0 — 2026-06-23 — Cross-platform Chroma enhancements

Duckets asked: can we enhance the Chroma DB? Make it work on Windows?
Push to main? Three concrete additions:

### New: `compact` CLI subcommand

- `python -m src.cli compact` — dedupes + VACUUMs the Chroma store.
  Real-world result on the existing 4084-chunk store: saved **10.4 MB**
  by vacuuming the SQLite WAL.
- Scans every tier collection for duplicate ids, keeps the most
  recently-ingested copy, re-upserts to overwrite the dupes.
- Runs `VACUUM` on the underlying `chroma.sqlite3` (cross-platform;
  Python's stdlib `sqlite3` module handles Win/Mac/Linux identically).
- Refuses to run on non-Chroma backends (Qdrant / LanceDB) with a
  clear error message.

### New: `distance_metric` knob on `ChromaBackend`

- Three options: `cosine` (default), `l2` (Euclidean), `ip` (inner
  product). `ip` is faster for pre-normalized vectors (BGE models
  with `normalize_embeddings=True`).
- Backed by `DUCKBOT_CHROMA_DISTANCE` env var.
- Threaded through `src/store.py:MemoryStore` → `get_backend()` →
  `ChromaBackend.__init__`.
- Chroma's `hnsw:space` only takes effect on collection CREATION, so
  changing the metric on an existing store requires a new persist
  dir or reset. Documented in the README.

### New: Windows support (scripts/secret-scan.ps1 + install-pre-commit.ps1)

- `scripts/secret-scan.ps1` — PowerShell port of `secret-scan.sh`.
  Same patterns, same logic, same exit codes. Works on Windows 10/11
  with PowerShell 5.1+ (ships with Win 10) and PowerShell 7+
  (cross-platform).
- `scripts/install-pre-commit.ps1` — installs the pre-commit hook
  on Windows. Auto-detects pwsh vs bash and installs the right shim.
- Both files are gitignored-from-committing-secrets via the
  `duckbot-secret-scan: allowlist-file` top-of-file marker.
- The bash version is still the default on macOS/Linux; symlink in
  `.git/hooks/pre-commit` already exists in the repo.

### README + cross-platform notes

- Added a "Cross-platform support" section to README.md covering
  macOS / Linux / Windows quirks (path limits, HF Hub auth, pwsh
  versions, Chroma wheels).
- Documented the new `compact` and `distance_metric` commands.

### Verification

- 419/419 tests pass (was 404; +15 from new test file).
- `compact` end-to-end on the real 4084-chunk store: 0 duplicates,
  10.4 MB saved.
- Doctor clean; secret-scan clean.
- All Python code is `pathlib`-based (no `os.path.join` literals),
  so the core works on Win/Mac/Linux identically.

## 0.7.0 — 2026-06-23 — Weighted RRF + FSRS-6 (L11 + L9)

Two more layers landed: per-tier prior weighting (L11) and the FSRS-6
spaced-repetition algorithm (L9). Both default OFF — L7 (cross-encoder
rerank), L8 (Ebbinghaus decay), and L13 (verbatim) remain the defaults.

### L11 — Weighted RRF with per-tier priors

- **`src/tier_priors.py`** — `maybe_apply_tier_priors()` multiplies each
  result's RRF by a per-tier weight. Defaults: procedural=1.5,
  semantic=1.2, episodic=1.0, working=0.8. Pattern from Cognee's
  tier-aware RRF (Apache-2.0) and MemPalace's per-section weight map
  (MIT). Audit fields (`_tier_prior`, `_rrf_score_pre_prior`) attached
  for downstream observability.
- Opt-in via `tier_priors=True` kwarg or `DUCKBOT_TIER_PRIORS=1`.
- Overridable per-call via `tier_priors_overrides={"procedural": 2.0}`.
- Threaded through `query.py` → `memory.py` → `connectors/base.py` →
  `connectors/openclaw.py` (gain `tier_priors` + `tier_priors_overrides`).
- 21 tests in `tests/test_tier_priors.py` covering defaults, opt-in
  dispatch, math correctness, real `QueryResult` round-trip.

### L9 — FSRS-6 spaced repetition math

- **`src/fsrs.py`** — reimplementation of the FSRS-6 algorithm spec
  (public-domain math, NOT from any source code):
  - `fsrs_retrievability(t, S) = (1 + t/(9S))^(-w20)` — AnKing form
    with default w20=0.9 (steeper than the published 0.1542 because
    our chunks are denser knowledge items).
  - `fsrs_bump_stability(S, D, R)` — success: `S' = S * (e^w8 * (11-D) * S^-0.8 * (1-R) + 1)`.
  - `fsrs_bump_difficulty(D, R)` — `D' = D - w6*(R-0.5)` on success,
    `D' = D + w6*(1-R)` on failure.
  - `maybe_fsrs()` — opt-in dispatch matching the L7/L8 pattern.
    Reads per-chunk `stability_days` + `difficulty` from metadata.
    Fallback to `last_recalled_at` → `created_at` → `ingested_at`
    for elapsed time.
- Opt-in via `fsrs=True` kwarg or `DUCKBOT_FSRS=1`.
- 41 tests in `tests/test_fsrs.py` covering R(t, S) power-law,
  stability growth under easy/hard difficulty, difficulty updates
  on success/failure, audit fields, env var dispatch, and the
  timestamp-fallback chain.

### Verification

- 404/404 tests pass (was 342 after L14; +21 L11 + +41 L9 = +62).
- End-to-end via `Brain.recall(rerank=True, tier_priors=True, fsrs=True)`:
  SOUL.md procedural rule wins with score 1.176 (boosted by both
  rerank and tier prior × retrievability).
- Secret-scan clean.

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
