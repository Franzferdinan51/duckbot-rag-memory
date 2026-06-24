# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.11.x  | ✅ Active          |
| 0.10.x  | ✅ Security patches |
| < 0.10  | ❌ End of life     |

We follow semver — security fixes ship to the latest minor version
and the previous minor version. Older versions get no backports.

## Reporting a Vulnerability

**Please do not open a public GitHub issue for security issues.**

This project handles API keys (LM Studio, OpenAI, MiniMax), wallet
addresses (PEARL mining), and personal notes — all of which are
sensitive if leaked. We treat reports confidentially and respond fast.

### How to report

1. **Email** (preferred for sensitive reports): see the GitHub
   profile of the maintainer `Franzferdinan51` for the contact email.
2. **GitHub Security Advisories** (alternative):
   https://github.com/Franzferdinan51/duckbot-rag-memory/security/advisories/new

Include in your report:
- The vulnerability description + steps to reproduce
- Affected versions
- Whether you've tested the impact / exploitation
- Suggested fix (if any)

### What to expect

- **Acknowledgment** within 48 hours
- **Status update** within 7 days
- **Fix + CVE** (if appropriate) within 30 days for critical issues,
  90 days for high/medium

We will credit you in the fix release unless you prefer to remain
anonymous.

## What we consider a security issue

- Credential leakage: API keys, wallet addresses, secrets in commits
  or in `.env` files that ship with the repo
- Path traversal / sandbox escape in the watcher daemon or MCP server
- Code execution from user-supplied input (chunk.py, embed prompts, etc.)
- Bypass of the secret-scan pre-commit hook
- Insecure defaults that leak data (e.g. embedding cache writing secrets)
- Dependency vulnerabilities with a known exploit

## What is NOT a security issue

- "The embed endpoint is slow" — performance, not security
- "The CLI prints a stack trace on bad input" — UX, not security
- "The MCP tool description is too long" — docs, not security
- Theoretical attacks without a concrete exploit path

## Hardening checklist (for users)

Before you `git clone` this on a new host:

- [ ] `.env` is in `.gitignore` (it is — verify `cat .gitignore | grep .env`)
- [ ] `git log --all -p | grep -E "LMSTUDIO_API_KEY=|OPENAI_API_KEY=|prl1[a-z0-9]{90}"` returns nothing
- [ ] Pre-commit hook installed: `pip install pre-commit && pre-commit install`
- [ ] LM Studio bound to `127.0.0.1`, not `0.0.0.0` (default — verify in LM Studio Developer settings)
- [ ] If using OpenAI: key has spend limits set in OpenAI dashboard
- [ ] If mining: PEARL wallet address is in `.env` (gitignored), not in
      any tracked file (we use `secret-scan` to enforce)

## Security-relevant code paths

If you're reviewing for security, these are the high-value targets:

- `src/embeddings.py` — `_get_http_client()`, `_rate_limiter` — verifies
  outbound HTTPS to embed providers, no arbitrary URL fetches
- `src/mcp_server.py` — stdio JSON-RPC server, no network listener
- `src/watcher.py` — polls filesystem; reads files in `.git`, `node_modules`,
  `.venv` etc. are excluded to avoid embedding irrelevant / sensitive
  content into the vector store
- `scripts/install.ps1` + `scripts/install.sh` — register a
  Task Scheduler / launchd / systemd service; review before running
- `scripts/secret-scan.{ps1,sh}` — pre-commit guard for secrets
