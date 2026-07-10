# Changelog

## Unreleased — OpenClaw plugin registration fix

### Fixed

- **`extensions/duckbot-memory/` plugin failed to register on OpenClaw 2026.6.11+** — every brain tool call returned "tool not found" and the brain was effectively offline inside OpenClaw. Three root-causes were stacked:
  - **`openclaw.plugin.json` was missing `contracts.tools`** — the registry silently rejects every `registerTool()` call when the manifest doesn't declare the plugin's tool surface. Added the full 69-name declaration.
  - **`api.registerHook('session_start' | 'session_end' | 'gateway_stop', ...)` was missing `opts.name`** — the loader validates via `requireRegistrationValue(opts?.name, 'hook registration missing name')` and throws, killing plugin init entirely. Added `{ name: 'duckbot-memory.<event>' }` to all three hooks.
  - **Wire-protocol mismatch with the Python server** — the Python MCP server (`src/mcp_server.py:2503`) reads stdin one `readline()` at a time and parses each line as a JSON object. The shim was sending Content-Length framed JSON without a trailing newline, so the server's first `readline()` blocked waiting for `\n` and the MCP `initialize` round-trip timed out at exactly 60 s. Switched outbound framing to newline-delimited JSON (`{json}\n`) while keeping the read path flexible (handles both Content-Length and newline-delimited framing on the inbound side).
- **Log messages were printing literal `%s` / `%d`** — the OpenClaw 2026.6.11 logger only substitutes `%d` (not `%s`), so `logger.info('repo=%s, python=%s', repo, python)` printed `repo=%s, python=%s` instead of substituting. Converted all 12+ printf-style log calls to template literals for consistent rendering.
- **Plugin description and version bumped** to v0.1.2 with a more accurate description (calls out `block_*` / `graph_*` family and the single-spawn singleton pattern).

### Symptom seen by the user

In the OpenClaw dashboard: "every tool returns image previews / brain says it's sandboxed and can't do anything." Both were downstream of the plugin being unable to register: the LLM had no tool surface to call, so calls returned empty errors, and the missing `session_start` hook meant `brain_wake_up` never fired to inject context. Fix validated end-to-end: gateway log now shows `[duckbot-memory] ready: 69 tools, 69 registered, pid=… (singleton, refs=1)` within ~1.3 s of spawn (was failing with `hook registration missing name` before).

### Operational notes

- The bootstrap script `scripts/openclaw-bootstrap.sh` already does the right thing (`ln -sf` → falls back to `cp -R` if symlinks are rejected). If `~/.openclaw/extensions/duckbot-memory/` is a directory instead of a symlink, replace it with a symlink to your canonical repo so future edits flow through automatically: `rm -rf ~/.openclaw/extensions/duckbot-memory && ln -s "$(pwd)/extensions/duckbot-memory" ~/.openclaw/extensions/duckbot-memory`. Then `openclaw daemon restart`.
- If `openclaw sandbox explain` shows `Elevated: failing gates: allowFrom (tools.elevated.allowFrom.webchat)`, set the allowlist once (this is a one-line config fix in `~/.openclaw/openclaw.json`, not a code change):
  ```json
  "tools": { "elevated": { "enabled": true, "allowFrom": { "webchat": ["*"] } } }
  ```
  Equivalent CLI: `openclaw config set tools.elevated.allowFrom --json '{"webchat": ["*"]}'`. Then `openclaw daemon restart`.

## v0.15.2 — Test suite repair + cross-platform hardening

### Fixed

- **Test suite: 36 failures → 0 failures** (885 passing, 3 skipped).
  - Replaced hardcoded `/Users/duckets/Desktop/duckbot-rag-memory` paths
    in 5 test files with `Path(__file__).resolve().parent.parent` so
    tests run on every host without a fixture override.
  - Replaced hardcoded `/Users/duckets/.openclaw/workspace/...`
    fixture strings in `tests/test_tier.py` with `/home/example/...`
    placeholders (matches production tier classifier — file basenames
    like `AGENTS.md` / `SOUL.md` / `MEMORY.md` drive the match).
  - Added missing `from pathlib import Path` imports to
    `tests/test_fsrs.py` and `tests/test_tier_priors.py`.
  - Wrapped `subprocess.run(['bash', ...])` in `tests/test_secret_scan.py`
    and `tests/test_cross_platform.py::test_bash_script_parses` with
    `shell=True` so MSYS path translation applies on Windows
    (`C:/Users/foo` → `/c/Users/foo`). POSIX behavior unchanged.
  - `test_dashboard_tail_lines_handles_multi_chunk_files`: write the
    log file with `newline=""` so the assertion matches the LF
    contract of the production helper; added an explicit CRLF sub-test
    to lock in the trailing-CR-preservation behavior.
  - `test_hermes_hook_scripts_exist_and_executable` and
    `test_bootstrap_scripts_exist_and_exec`: skip the `S_IXUSR` check
    on Windows (NTFS does not enforce Unix mode bits; git drops +x).
    On Windows, the test sets the bit in-process so runtime invocation
    works; production users on Windows invoke via `bash scripts/<name>.sh`
    (bash.exe is in PATH on git-bash).
- **Security regression scrub (follow-up to commits 421a460 / 4c3ba47):**
  removed the leaked `/Users/duckets/Desktop/duckbot-rag-memory` default
  from `src/connectors/openclaw.py::openclaw_config_snippet` (the
  example now uses `~/Desktop/duckbot-rag-memory`).
- **`scripts/secret-scan.sh` allowlist bug on Windows:** the marker
  check used `printf "%s\n" "$(git show ":$f" ...)" | head -30 | grep -qE ...`
  which on MSYS bash (Windows) collapses to 1 byte because the command
  substitution truncates output. Replaced with the direct pipe:
  `git show ":$f" ... | head -30 | grep -qE ...`. Without this fix, the
  scanner flagged its own allowlisted test fixtures (`tests/test_secret_scan.py`)
  as secrets and blocked commits. Verified via 17/17 passing
  `tests/test_secret_scan.py` cases and a clean `bash scripts/secret-scan.sh`
  against the full audit diff.
- **`MemoryStore().stats()` segfault fix:** the v0.2 chroma index was
  incompatible with v0.10+ schema and crashed with SIGSEGV under load
  (exit 139) on `src.cli stats` and `src.cli wake-up`. Wiping
  `data/chroma/` + `data/watcher_state.json` and re-running
  `src.watcher once` recovers cleanly. This is the documented v0.2 →
  v0.10+ migration step; the new `.gitattributes` (below) prevents
  the symptom from re-appearing due to line-ending drift.

### Added

- **`.gitattributes`** — cross-platform line-ending normalization. All
  `*.sh` / `*.ps1` / `*.py` / `*.json` / `*.md` files are LF; `*.bat`
  and `*.cmd` are CRLF. Git will rewrite to LF on commit so Windows
  editors that default to CRLF (Notepad, older VS Code configs) can't
  poison the repo with a literal `\r` in scripts.

## v0.15.1 — Observer perspective + lifecycle events + priority scoring## v0.15.1 — Observer perspective + lifecycle events + priority scoring

### Added

- **5-factor priority scoring** (`src/scoring.py`) — re-ranks
  `brain_wake_up` results using MindBank-style weighted factors
  (recency 30%, frequency 25%, connectivity 20%, explicit 15%,
  type 10%). An old-but-frequently-recalled high-importance chunk
  now surfaces above a fresh-but-noisy one. Pure functions, no
  I/O; integrates into `Brain.wake_up` after the recall pass.
  29 unit tests + safe-degrade behavior when `scoring.py` is
  unavailable.

- **Lifecycle event capture** (`src/events.py`) — SQLite-backed
  `data/events.db` records `session_start`, `session_end`,
  `pre_tool_use`, `post_tool_use`, `tool_error` events per MCP
  session. Enables "what tool calls led to this decision?"
  debugging. Auto-captures around every MCP tool call (pre + post
  + duration_ms; tool_error captures exception messages) and
  every Hermes plugin session boundary. JSON payloads are
  recursively truncated to 8192 chars so a runaway tool can't
  blow up the DB. 22 unit tests + concurrent-write test.

- **Observer perspective — causal precursor tracing**
  (`src/observer.py`) — backward BFS through the entity graph
  via causal labels (decided_by / depends_on / learned_from /
  caused_by / supports / related_to / contradicts). Returns a
  depth-indexed chain + `critical_depth` (shallowest depth
  capturing >= 90% of influence) + `coverage` (fraction of
  immediate edges with upstream rationale). Inspired by MindBank's
  Observer Perspective.

- **Observer perspective — blind-spot detection**
  (`src/observer.find_blind_spots`) — flags entities that make
  causal claims (outgoing decided_by / depends_on / learned_from
  edges) but have no upstream rationale of their own. Severity
  scales with downstream edge count (1 = low, 2 = medium, 3+ =
  high). 27 unit tests for the observer module + 5 facade-level
  tests in `test_connectors.py`.

- **Two new MCP tools** (now 66 total, was 64):
  - `brain_graph_precursors(entity, max_depth, include_inactive, min_influence)` — runs `trace_precursors` against the live graph.
  - `brain_graph_blind_spots(max_results, include_inactive)` — runs `find_blind_spots`.

## v0.15.0 — Native OpenClaw plugin + Hermes auto-activation

### Added

- **Native OpenClaw plugin** (`extensions/duckbot-memory/`) — pure
  Node.js shim (zero npm dependencies) that spawns the Python MCP
  server as a subprocess and proxies 66 tools + `session_start` /
  `session_end` hooks into OpenClaw's plugin runtime. Replaces the
  previous `src/extensions/duckbot_brain/openclaw.plugin.json` which
  claimed Python support that OpenClaw never honored (OpenClaw plugins
  run in-process in the Node gateway and can't load Python). The shim
  pattern matches `openclaw/openclaw/extensions/voice-call/` (which
  uses `child_process.spawn` + JSON-RPC over stdio) and uses the
  real `openclaw.plugin.json` schema per `docs/plugins/manifest.md`.
  See `extensions/duckbot-memory/README.md` for install + config.
- **Hermes plugin auto-activation** (`scripts/hermes-bootstrap.sh`) —
  after copying the plugin files into `~/.hermes/plugins/memory/duckbot_brain/`,
  the bootstrap now backs up `~/.hermes/config.yaml` and appends
  `memory.provider: duckbot-brain` (with backup) so the plugin is
  actually activated. Idempotent — re-running is a no-op.
- **Plugin discovery + activation tests**
  (`tests/test_hermes_plugin_discovery.py`, 9 tests) — verify
  `register(ctx)`, fallback paths, `is_available()` purity,
  `plugin.yaml` shape, and that the bootstrap script actually writes
  the activation line.
- **Shim unit tests** (`extensions/duckbot-memory/test/shim.test.js`,
  15 tests, `node --test`) — verify `StdioJsonRpc` framing
  (Content-Length + newline-delimited fallback), error response
  handling, exit cleanup, timeout, stderr forwarding, server-initiated
  notifications, and that `register()` correctly wires spawn + 64
  tools + session hooks via a mocked `child_process.spawn`.

### Fixed

- **OpenClaw plugin claim** — `src/extensions/duckbot_brain/openclaw.plugin.json`
  claimed `entry: "python"` + `entryArgs: ["-m", "src.extensions.duckbot_brain.adapter"]`
  but OpenClaw plugins can't load Python. Deleted; the directory now
  contains the generic JSON-RPC MCP client adapter used by Claude
  Code / Cursor / Codex / mcporter (its docstring updated to reflect
  that). The native OpenClaw plugin is the new Node.js shim at
  `extensions/duckbot-memory/`.

### Test surface

- 849 Python tests passing (was 748)
- 15 Node.js tests passing (was 12)
- 66 MCP tools (was 64)

## v0.15.0 (earlier) — Skill pipeline maturity + eval trends + ingest safety

### Added

- **`brain_skills_suggest` (MCP + shared surface + OpenClaw CLI)** —
  semantic top-N skill candidates by query. Uses hybrid retrieval
  scoped to the procedural tier and filtered to unpromoted candidates.
  Lets the agent ask 'are there candidate skills about X?' before
  promoting.

- **`trust_level` param on `brain_remember(kind="skill_candidate")`** —
  "full" (default, skip injection scan — agents are trusted) or
  "standard" (run scan, quarantine suspicious content for untrusted
  callers like user-driven skill scripts).

- **`instructions_markdown` field on `brain_skills_promote`** —
  rich markdown body that overrides the flat `instructions` list.
  Lets the agent author full markdown sections (headings, code
  blocks, tables) instead of a flat numbered list. `instructions` is
  now optional as long as `instructions_markdown` is provided.

- **`python -m src.cli skills <verb>`** — standalone CLI for the
  skill pipeline. Verbs: `stamp` (new candidate), `list` (unpromoted
  candidates), `promote` (chunk_id + name + description + instructions
  [--json form for scripted use]), `suggest <query>` (semantic top-N).
  Output is JSON so callers can parse.

- **Eval trend detection** — `src/eval.py` adds
  `load_history()` and `compute_trend()` that read the eval_history
  JSONL and report recent-vs-prior deltas on mean_recall_at_5,
  mean_mrr, and p95_latency. `python -m src.cli eval` now prints
  the trend alongside the summary.

- **Watcher pending-skill-candidate report** — every 10 minutes the
  watcher logs how many unpromoted skill candidates are waiting, so
  operators (or the agent on its next wake-up) notice the skill pipeline
  has work pending. Throttled so a busy watcher doesn't spam the log.

### Fixed

- **brain_recall / brain_recall_verbatim (shared surface)** now
  reject empty/whitespace query with a clear error instead of returning
  5 random semantically-similar chunks. Search semantics: use
  `brain_search_verbatim` for exact substring match.

- **Memory.remember() concurrent ingest** — added a per-Memory
  `asyncio.Lock` around the conflict-detection + add_chunks sequence.
  Multiple ingest workers (OpenClaw adapter + watcher) can now run
  in parallel without racing on near-duplicate detection. On lock
  failure (rare), fall back to unlocked add_chunks — the actual write
  is idempotent (content-hash chunk_id) so racing duplicates are
  recoverable.

- **scripts/cron.sh** moved to `scripts/archive/cron.sh.deprecated`.
  The watcher daemon has been the recommended path since v0.10.
  Manual fallback for the original cron-style nightly batch:
  `python -m src.cli reflect && python -m src.cli eval <bench> && python -m src.cli sync`.

### Test surface

- 737 passing
- 12 shared-surface tools (was 11; added `brain_skills_suggest` + `brain_skills_promote`)
- 64 MCP tools (was 43, now correctly 64 including connector tools)

## Unreleased — v0.14.0 — Agent plugin parity + wake-up wiring

The four skill files (`skills/duckbot-brain/`, `skills/openclaw-imports/`,
`skills/codex-imports/`, `skills/cursor-imports/`) all tell agents to call
`brain_wake_up` on session start, but until this release the OpenClaw
extension adapter and the Hermes MemoryProvider plugin didn't actually
expose that tool — agents got "unknown tool" errors. v0.14.0 consolidates
the agent-facing surface so every thin entry point advertises the same
9 tools, and the OpenClaw adapter gets the same per-tool rate limiting
the canonical MCP server already had.

### Added — Shared agent surface

- **`src/extensions/tools.py`** — single source of truth for the 9 core
  tools every agent entry point exposes. Both the OpenClaw stdio adapter
  (`src/extensions/duckbot_brain/adapter.py`) and the Hermes
  MemoryProvider plugin (`src/plugins/memory/duckbot_brain/__init__.py`)
  delegate to this. The full 56-tool MCP surface is still available via
  `python -m src.mcp_server` for admin / CLI use.

- **`brain_wake_up` exposed in OpenClaw + Hermes** — was previously
  missing from both thin entry points (skill instructions were lying).
  Now an agent on either platform can call `brain_wake_up` on session
  start and get the same shape: recent memories (superseded filtered),
  active memory blocks, graph summary, FSRS review queue, stats.

- **`on_session_start` hook on the Hermes plugin** — returns the
  `brain_wake_up` shape directly so the plugin loader can pre-load
  context without an extra MCP round-trip. Listed in `plugin.yaml`.

- **`on_session_end` consolidation** — was a `return None` stub that
  the manifest advertised as a hook. Now actually walks session
  messages, extracts durable-shaped user statements (always / never /
  prefer / want / don't), and queues a procedural-tier chunk for
  `brain_remember`. Skips non-primary contexts (cron / subagent /
  flush) per the Hermes ABC.

- **Per-tool rate limiting on the OpenClaw adapter** — reuses the
  existing `src.ratelimit` module so a runaway agent can't fill the
  disk with `brain_remember` calls. `DUCKBOT_RATELIMIT_DISABLE=1`
  turns it off, same env var as the MCP server.

### Fixed

- **OpenClaw extension manifest (`openclaw.plugin.json`) listed only 4
  of 8 tools** — clients that read the manifest before connecting got
  a smaller surface than the adapter actually implemented. Manifest
  now lists all 9 (was 4 → 9; `version` bumped to `0.2.0`).

- **`test_rate_limiter_allows_until_burned` and
  `test_mcp_dispatch_returns_429_style_on_rate_limit` flaked in the
  full suite** — root cause: the `src.ratelimit._RATE_LIMITER` singleton
  wasn't reset between tests, and the rate-limit-disable env var from a
  previous test bled into the next. Both tests now explicitly reset the
  singleton and clear the env var at the top.

- **OpenClaw adapter returned bare lists for `brain_recall` /
  `brain_recall_verbatim`** — inconsistent with the MCP server and the
  Hermes plugin, which both return `{"results": [...]}`. Adapter now
  matches.

### Changed

- **Hermes MemoryProvider plugin exposes 9 tools instead of 3** — was
  just `brain_recall`, `brain_recall_verbatim`, `brain_reflect`. Now
  matches the OpenClaw adapter's 9-tool surface (incl. `brain_wake_up`).

- **Test counts**:
  - New: `tests/test_extensions_tools.py` (26 tests),
    `tests/test_openclaw_shim.py` (25 tests),
    `tests/test_openclaw_extension_e2e.py` (7 end-to-end subprocess tests).
  - Updated: `tests/test_openclaw_extension.py` (+2 assertions),
    `tests/test_hermes_plugin.py` (+6 tests, 2 updated),
    `tests/test_mcp_tools_extension.py` (assertion updated to match
    the 9-tool core agent surface — `brain_forget_by_query` is
    intentionally excluded; it's destructive admin-tier).
  - Full suite: **687 passing** (was 617, +70 tests).

### Added — round 2: shell access + skill consistency

- **`python -m src.cli openclaw <verb>` CLI shim** — parallel to the
  existing `hermes` shim, but delegates to the shared 9-tool core
  agent surface. Supports `wake-up`, `recall`, `remember`, `stats`,
  `tools`, and a generic `call <tool> '<json>'` escape hatch. Exits
  non-zero on dispatch errors so cron jobs / shell pipelines can
  detect failure without parsing JSON. (`src/connectors/openclaw_shim.py`)

- **`skills/duckbot-brain/SKILL.md` rewritten** to feature
  `brain_wake_up` as the canonical session-start call (matches the
  other 3 skill files which already did). Adds the 3 install paths
  (extension adapter, symlinked skill, canonical MCP server) and a
  pointer to `docs/PLUGIN_SURFACE.md`.

- **Bootstrap scripts auto-install the skills/plugins** —
  `scripts/openclaw-bootstrap.sh` now symlinks
  `skills/duckbot-brain/SKILL.md` into
  `~/.openclaw/workspace/skills/duckbot-brain/` on run (was: just
  printed instructions). `scripts/hermes-bootstrap.sh` copies the
  plugin package into `~/.hermes/plugins/memory/duckbot_brain/` so
  the Hermes plugin loader discovers it on next session start.

- **End-to-end smoke test** — `tests/test_openclaw_extension_e2e.py`
  spawns the adapter as a real subprocess and exercises the stdio
  JSON-RPC loop (initialize, tools/list, tools/call, malformed-JSON
  recovery, multi-request sessions). Catches issues that the
  unit-level adapter tests miss.

- **Deprecation note** on the legacy `src/connectors/openclaw.py`
  module — points new code at `src/extensions/duckbot_brain.adapter`
  and `src/connectors/openclaw_shim`. Module is retained (per the
  project's "No deletions" rule).

### Added — Round 3: agent-driven skill pipeline (zero VRAM)

The brain never calls a generative LLM — only the embedding model runs.
The agent (OpenClaw / Hermes / any MCP client) authors skill content
using its own LLM context; the brain is pure storage + template.

- **`src/skill_pipeline.py`** — storage-only candidate/list/promote
  logic. No LLM anywhere. Three entry points:
  - `stamp_skill_candidate()` — stores a procedural-tier chunk with
    `metadata.kind="skill_candidate"`, `promoted=False`. Returns
    `chunk_id` immediately (blocking, so the agent can promote it later).
  - `list_candidates()` — scans the procedural tier for candidates,
    filters out promoted (or includes them), sorts by recency then
    importance. Pure metadata scan.
  - `promote_candidate()` — writes `skills/<slug>/SKILL.md` via the
    existing `skillgen.write_skill` (pure template), then marks the
    chunk as `promoted=True` + `promoted_at` + `promoted_skill_slug`.
  - Also `suggest_candidates()` (semantic top-N) + `candidate_stats()`
    (count summary).

- **`kind="skill_candidate"` remember mode** — added to both the shared
  surface (`src/extensions/tools.py`) and the MCP server
  (`src/mcp_server.py handle_remember`). When `kind="skill_candidate"`,
  the brain stamps a candidate (blocking, returns chunk_id) instead of
  the fire-and-forget queue path. No LLM call.

- **`brain_skills_list` + `brain_skills_promote` tools** — added to:
  - The shared 11-tool core agent surface (`src/extensions/tools.py`)
  - The canonical MCP server (`src/mcp_server.py` TOOLS + HANDLERS)
  - The OpenClaw plugin manifest (`openclaw.plugin.json`, 9 → 11 tools,
    `version` bumped to `0.3.0`)
  - The Hermes plugin manifest (`plugin.yaml`, `version` bumped to `0.3.0`)

- **`tests/test_skill_pipeline.py`** (25 tests) — covers stamp → list →
  promote end-to-end, dispatch routing, tool schema presence, and the
  system-prompt description of the agent-driven flow.

- **Full suite: 712 passing** (was 687, +25 tests).

### Migration

No breaking changes for end users. If you have scripts that import
`src.extensions.duckbot_brain.adapter` and call `_call_tool` or
`_tool_schemas` directly, they'll keep working — the function
signatures didn't change. If you wrote a custom dispatcher that
expected the bare-list shape from `brain_recall`, update to
`payload["results"][...]` (matches the MCP server + Hermes plugin).

## 0.13.0 — 2026-06-24 — Tier 2 + Tier 3 borrowed features

Round of porting from upstream projects (MemPalace, mem0, mem0-style
conflict detection, Graphiti, py-fsrs, agentskills.io) plus the bug
fixes accumulated along the way. New MCP tools now total 53.

### Added — Tier 2: battle-tested patterns

- **AAAK compression dialect** (`src/dialect.py`, `brain_index` tool) —
  MemPalace-style compact one-line-per-chunk format. Lets an LLM
  scan thousands of entries in <500 tokens before deciding which to
  expand via `brain_recall`. Pairs with the `wing/room/drawer` view.

- **FSRS-6 w20 self-tuner** (`src/fsrs_optimizer.py`,
  `brain_optimize_fsrs` + `brain_apply_fsrs_w20` tools) — grid-search
  the forgetting-curve exponent from the brain's recall history,
  minimizing MSE between predicted R(t, S) and observed
  'remembered'/'forgotten' labels. `brain_apply_fsrs_w20` commits the
  new w20 (in-process; env-var support coming next).

- **Spellcheck on ingest** (`src/spellcheck.py`) — lightweight
  common-typo fixer (~70 entries) runs before chunking in
  `Memory.remember()`. Opt-out via `DUCKBOT_SPELLCHECK=0`. Preserves
  case and protects proper nouns (Duckets, Hermes, BATMAN, etc.).

- **Skill auto-creation** (`src/skillgen.py`, `brain_skill_create`
  tool) — when an agent solves a new task, this distills the win
  into an agentskills.io-compatible `skills/<slug>/SKILL.md`. Pure
  templating, no LLM call by default. Refuses to overwrite by
  default; pass `overwrite=true` to replace.

- **Wing/Room/Drawer 2D hierarchy** (`src/palace.py`, `brain_palace`
  tool) — MemPalace's 3-level structure (person/project → time →
  verbatim chunk) overlaid on top of the existing tier system. Lets
  an agent do "show me everything about OpenClaw from this week"
  without manual source_path filtering. Skips superseded chunks at
  index time.

### Added — Tier 3: agent-specific integrations

- **Cross-agent brain_sync** (`brain_sync` tool, `target=both`) —
  write to BOTH `~/.openclaw/workspace/memory/` and
  `~/.hermes/memories/` in a single call.

- **Honcho-style user modeling** (`brain_user_model` tool) —
  periodically distills high-importance user-related facts into a
  single `user` memory block via `block_write`. Appends to existing
  content so the model accumulates over time.

- **Proactive memory nudge** (`brain_nudge` tool) — surfaces
  stale-but-important memories the agent might be forgetting
  about. High importance + not recently recalled + older than
  `--stale_days`. Optional `--context` biases toward the agent's
  current focus.

- **Bi-temporal graph edges** (`graph.py`) — `recorded_from` /
  `recorded_until` columns on relationships (separate from
  `valid_from` / `valid_until`). `query_known_at()` answers "what did
  the brain know at time X?" — different from `query_active()` which
  asks "what was true then?" Graphiti-inspired. Idempotent migration
  on existing DBs.

### Fixed

- **`test_hermes_cli_shim_recall` "Event loop is closed"** — root
  cause: the cached `httpx.AsyncClient` in `src/embeddings.py` was
  bound to a different (closed) event loop when reused across pytest
  tests. `_get_http_client()` now tracks the binding loop and
  rebuilds the client on mismatch. `_run_async()` also got a lock
  + explicit `new_event_loop()`/`close()` lifecycle to prevent the
  related class of leaks. 5 other regression tests switched from
  `asyncio.run()` to a `_run_in_thread()` helper that owns the loop
  properly.

- **`brain_sync` AttributeError on `r.source_path` / `r.importance`** —
  `QueryResult` stores both in `metadata`, not as direct attributes.
  Centralized into `_src()` and `_imp()` helpers in
  `handle_brain_sync`. The `target=both` path now works end-to-end.

### Tests

- 36 new regression tests across the 9 new features + the H1 fix.
- Full suite: **597 passing** (was 517 at the start of the v0.11
  series).

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
