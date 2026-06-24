# install-pre-commit.ps1 — install the secret-scan pre-commit hook on Windows.
#
# duckbot-secret-scan: allowlist-file
#
# Run from the repo root in PowerShell:
#   pwsh scripts/install-pre-commit.ps1
#
# What it does:
#   1. Creates .git/hooks/pre-commit (a tiny shim that calls our PS1).
#   2. Skips if already installed.
#
# The shim works on:
#   - Windows + Git Bash (most common)
#   - Windows + WSL
#   - Any POSIX shell (falls back to bash + .sh version)
#
# Idempotent: safe to re-run.

[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$RepoRoot = git rev-parse --show-toplevel 2>$null
if (-not $RepoRoot) {
    Write-Error "Not in a git repo. cd to the repo root and re-run."
    exit 1
}
Set-Location $RepoRoot

$HookDir = Join-Path $RepoRoot '.git/hooks'
$HookPath = Join-Path $HookDir 'pre-commit'
$Ps1Path = Join-Path $RepoRoot 'scripts/secret-scan.ps1'
$ShPath = Join-Path $RepoRoot 'scripts/secret-scan.sh'

if (-not (Test-Path $HookDir)) {
    Write-Error ".git/hooks does not exist. Is this a git repo?"
    exit 1
}

if ((Test-Path $HookPath) -and (Select-String -Path $HookPath -Pattern 'secret-scan' -SimpleMatch -Quiet)) {
    Write-Host "✓ pre-commit hook already installed at $HookPath"
    exit 0
}

# Determine the best shim for this environment.
$HasPwsh = $null -ne (Get-Command pwsh -ErrorAction SilentlyContinue)
$HasBash = $null -ne (Get-Command bash -ErrorAction SilentlyContinue)
$HasSh = $null -ne (Get-Command sh -ErrorAction SilentlyContinue)

if ($HasPwsh) {
    # PowerShell available — use the .ps1 version.
    $Shim = @"
#!/usr/bin/env pwsh
# Installed by scripts/install-pre-commit.ps1 — secret-scan.ps1 wrapper.
# duckbot-secret-scan: allowlist-file
`$ErrorActionPreference = 'Stop'
`$repoRoot = git rev-parse --show-toplevel 2>`$null
if (`$repoRoot) { Set-Location `$repoRoot }
& pwsh `"$Ps1Path`" `"`$args`"
exit `$LASTEXITCODE
"@
    Write-Host "✓ Installing PowerShell-based pre-commit hook (pwsh detected)..."
} elseif ($HasBash -or $HasSh) {
    # Fall back to bash + .sh version.
    if (-not (Test-Path $ShPath)) {
        Write-Error "No PowerShell AND no bash AND no sh. Cannot install hook."
        exit 1
    }
    $Shim = @"
#!/usr/bin/env bash
# Installed by scripts/install-pre-commit.ps1 — secret-scan.sh wrapper.
# duckbot-secret-scan: allowlist-file
exec bash "$ShPath" "`$@"
"@
    Write-Host "✓ Installing bash-based pre-commit hook (no pwsh found)..."
} else {
    Write-Error "No PowerShell, bash, or sh found. Cannot install hook."
    exit 1
}

Set-Content -Path $HookPath -Value $Shim -NoNewline
Write-Host "✓ Pre-commit hook installed at $HookPath"
Write-Host ""
Write-Host "Test it (should pass with no output):"
Write-Host "  git commit --allow-empty -m 'test hook'"
