<!--
Thanks for the PR! Fill in the sections below so reviewers know what to look at.
This project is open-source + cross-platform — please verify your change works on
at least one of (Windows / macOS / Linux). If you can only test one, note it.
-->

## What does this PR do?

<!-- One paragraph summary. -->

## Why?

<!-- What problem does this solve? Link the issue if there is one. -->

## How did you verify it?

<!--
- [ ] Tests pass locally (`pytest -v`)
- [ ] Manual smoke test on: ___ (Windows / macOS / Linux)
- [ ] If MCP/CLI surface changed: showed the new output here
-->

## Checklist

- [ ] No secrets committed (run `pwsh scripts/secret-scan.ps1` or `bash scripts/secret-scan.sh`)
- [ ] No deletions — only fixes or enhancements (project policy)
- [ ] Cross-platform: works on at least one of (Windows / macOS / Linux)
- [ ] If adding a CLI / MCP tool / API: documented in README.md, ARCHITECTURE.md, or INTEGRATION.md
- [ ] If adding a Python dep: added to `requirements.txt`
- [ ] Added/updated test(s) under `tests/`
- [ ] CHANGELOG.md updated (under "Unreleased" if not yet tagged)
