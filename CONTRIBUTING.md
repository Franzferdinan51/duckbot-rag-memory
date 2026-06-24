# Contributing

Thanks for taking an interest. This project is small, well-organized,
and welcoming to first-time contributors.

## TL;DR

1. Fork → branch → commit → push → PR
2. Read the PR template (auto-pops on new PR)
3. Run `pytest -v` + `bash scripts/secret-scan.sh` before pushing
4. Update `CHANGELOG.md` (under "Unreleased") + tests
5. Wait for review. Small PRs merge fast.

## Project values (in priority order)

1. **Open-source + community-first.** Every external dep + every design
   pattern traces back to a public project (mem0, Letta, Cognee, Hermes
   Agent, CoALA paper, etc.). See `docs/RESEARCH.md` for the lineage.
2. **Cross-platform.** Windows / macOS / Linux must all work. If you can
   only test on one, say so in the PR — we'll cover the rest.
3. **No secrets.** API keys, wallet addresses, private paths — never
   commit them. The `secret-scan` pre-commit hook catches them before
   they're pushed. `.env` is gitignored.
4. **No deletions.** Additive changes only. Refactors that rename
   things need a migration path.
5. **Honest limitations.** Document bugs, don't paper over them. Inline
   `# TODO` comments and CHANGELOG entries are welcome.
6. **Tested.** New features ship with tests. Bug fixes ship with
   regression tests. CI runs on push.

## Repo layout

```
duckbot-rag-memory/
├── .github/             # Issue/PR templates + GitHub Actions
├── benchmarks/          # golden.jsonl for `python -m src.cli eval`
├── data/                # gitignored: chroma db + watcher state + logs
├── docs/                # ARCHITECTURE, INTEGRATION, RESEARCH
├── scripts/             # install / start / cron / launchers
│   ├── install.{ps1,sh,linux.sh,macos.sh}    # bootstrap
│   ├── start-watcher.{ps1,sh,windows.bat}    # file-watcher daemon
│   ├── duckbot-memory-mcp.{sh,bat}            # MCP stdio launcher
│   ├── duckbot-ask, brain-recall.sh           # shell brain-query helpers
│   ├── secret-scan.{ps1,sh}                  # pre-commit secret guard
│   └── _format_{snippet,compact}.py           # format helpers for duckbot-ask
├── skills/              # OpenClaw skill manifests + plugins
├── src/                 # core library
│   ├── chunk.py embeddings.py tier.py store.py query.py consolidate.py eval.py cli.py
│   ├── memory.py        # the Memory facade
│   ├── mcp_server.py    # stdio JSON-RPC server
│   ├── watcher.py       # polling daemon
│   ├── decay.py fsrs.py rerank.py tier_priors.py  # brain layers
│   ├── blocks.py entities.py graph.py              # brain layers
│   ├── connectors/      # OpenClaw, Active Memory, dreaming, learn
│   └── backends/        # chroma / lancedb / qdrant (pluggable)
├── tests/               # pytest (CPU-only CI; LM Studio tests skip)
├── AGENTS.md            # quickstart for AI agents
├── ARCHITECTURE.md      # (in docs/) deep dive
├── CHANGELOG.md         # version history
├── CONTRIBUTING.md      # this file
├── INTEGRATION.md       # (in docs/) hermes/openclaw wiring
├── LICENSE              # MIT
├── README.md            # one-pager
├── SECURITY.md          # vuln disclosure policy
├── pytest.ini           # test config
├── requirements.txt     # pip deps
└── .pre-commit-config.yaml  # local secret-scan hook
```

## Dev setup

```bash
git clone https://github.com/Franzferdinan51/duckbot-rag-memory.git
cd duckbot-rag-memory
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env        # then edit to add your LM Studio / OpenAI key

# Optional but recommended:
pip install pre-commit
pre-commit install          # enables the secret-scan hook

# Run tests
pytest -v
```

If you don't have an LM Studio or OpenAI key, tests will skip the
integration tier but still pass — the unit tests cover the full
brain logic via mocks.

## Coding conventions

- **Python 3.11+ syntax.** `from __future__ import annotations` at top
  of every module. Type hints on public functions.
- **Cross-platform first.** No hardcoded `C:\` paths. Use `pathlib.Path`.
  Scripts use shebangs (`#!/usr/bin/env bash`, `.ps1`) and run on
  PowerShell 5+ + git-bash + bash + zsh.
- **No silent failures.** `try/except` should at minimum `log()`
  the failure, never bare `except: pass`.
- **Docstrings on public functions.** One-line summary is fine.
- **Commits**: `type(scope): summary` format, e.g.
  `feat(brain): v0.11.0 OpenClaw dreaming bridge`. Past-tense body.
- **CHANGELOG entry per merge.** Add a line under `## Unreleased`
  before opening the PR.

## What we won't merge

- Anything that needs to delete existing functionality without a
  migration path (project policy)
- Hardcoded API keys, wallet addresses, personal paths
- Platform-specific code in `src/` (use `scripts/` for that)
- New external dependencies without justification (we're a small
  project; deps are a real cost)
- Untested new features (bug fixes without tests are fine)
- Changes to `data/` (gitignored — never commit it)
- "Drive-by" formatting / whitespace fixes (one PR per concern)

## Review process

1. Maintainer (`Franzferdinan51` or a delegated bot) reviews within 1-3 days
2. Small fixes (< 30 lines, no API change): fast-track
3. Anything bigger: at least one round of review
4. Squash-merge to `main`, with the PR description as the commit body

## Getting help

- **General questions**: open a [Discussion](https://github.com/Franzferdinan51/duckbot-rag-memory/discussions)
- **Bug reports**: [Issue template](.github/ISSUE_TEMPLATE/bug_report.yml)
- **Feature requests**: [Feature template](.github/ISSUE_TEMPLATE/feature_request.yml)
- **Security**: [SECURITY.md](SECURITY.md) (private)

## License

By contributing, you agree your contributions will be licensed under MIT.
See [LICENSE](LICENSE).
