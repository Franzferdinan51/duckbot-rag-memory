# Changelog

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
