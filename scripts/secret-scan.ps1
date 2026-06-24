# secret-scan.ps1 — pre-commit guard for the DuckBot brain repo (Windows).
#
# duckbot-secret-scan: allowlist-file
#
# PowerShell port of scripts/secret-scan.sh. Same patterns, same logic,
# same exit codes. Works on Windows 10/11 with PowerShell 5.1+ (ships
# with Windows 10) and PowerShell 7+ (cross-platform).
#
# Pattern source: MemPalace's `.pre-commit-config.yaml` (MIT).
# https://github.com/MemPalace/mempalace/blob/develop/.pre-commit-config.yaml
#
# Install on Windows (from repo root in Git Bash or PowerShell):
#   # Either symlink (requires admin or Developer Mode):
#   New-Item -ItemType SymbolicLink -Path .git/hooks/pre-commit.ps1 -Target ../../scripts/secret-scan.ps1
#   # Or copy:
#   Copy-Item scripts/secret-scan.ps1 .git/hooks/pre-commit.ps1
#   # Then in .git/hooks/pre-commit, call this:
#   #   pwsh .git/hooks/pre-commit.ps1
#   # (Bash hook on Windows + WSL is also fine; just symlink the .sh version.)
#
# Skip with $env:DUCKBOT_SKIP_SECRET_SCAN=1 ONLY if you know what you're doing.

[CmdletBinding()]
param(
    [switch]$NoExit
)

$ErrorActionPreference = 'Stop'

# --- 1. Resolve repo root and opt-out ----------------------------------------

try {
    $RepoRoot = git rev-parse --show-toplevel 2>$null
} catch {
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $RepoRoot = Resolve-Path "$ScriptDir/.."
}
if (-not $RepoRoot) {
    $ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
    $RepoRoot = Resolve-Path "$ScriptDir/.."
}
Set-Location $RepoRoot

if ($env:DUCKBOT_SKIP_SECRET_SCAN -eq '1') {
    Write-Warning "WARNING: DUCKBOT_SKIP_SECRET_SCAN=1 — secret scan skipped"
    exit 0
}

# --- 2. Get staged files ----------------------------------------------------

$StagedFiles = @()
try {
    $StagedRaw = git diff --cached --name-only --diff-filter=ACM 2>$null
    if ($StagedRaw) {
        $StagedFiles = $StagedRaw -split "`n" | Where-Object { $_ } | ForEach-Object { $_.Trim() }
    }
} catch {
    # git not on PATH or not in a repo — treat as no staged files
    exit 0
}

if ($StagedFiles.Count -eq 0) {
    exit 0
}

# --- 3. Secret patterns -----------------------------------------------------
# Each: @{ Label = "..."; Regex = "..." }

$SecretPatterns = @(
    @{ Label = "OpenAI API key";     Regex = 'sk-[A-Za-z0-9]{20,}' },
    @{ Label = "Anthropic API key";  Regex = 'sk-ant-[A-Za-z0-9_-]{20,}' },
    @{ Label = "GitHub PAT";         Regex = 'ghp_[A-Za-z0-9]{36}' },
    @{ Label = "GitHub fine-grained"; Regex = 'github_pat_[A-Za-z0-9_]{60,}' },
    @{ Label = "AWS access key";     Regex = 'AKIA[0-9A-Z]{16}' },
    @{ Label = "MiniMax API key";    Regex = 'MiniMax-[A-Za-z0-9]{20,}' },
    @{ Label = "Bearer token";       Regex = 'Bearer\s+[A-Za-z0-9._-]{20,}' },
    @{ Label = "Generic high-entropy secret"; Regex = '(api[_-]?key|secret|token|password)\s*[:=]\s*["''][A-Za-z0-9._/+-]{16,}["'']' },
    @{ Label = "Private RSA key";    Regex = '-----BEGIN RSA PRIVATE KEY-----' },
    @{ Label = "Private OpenSSH key"; Regex = '-----BEGIN OPENSSH PRIVATE KEY-----' },
    @{ Label = "Private PGP key";    Regex = '-----BEGIN PGP PRIVATE KEY BLOCK-----' }
)

$Found = $false

# --- 4. Scan each staged file -----------------------------------------------

foreach ($f in $StagedFiles) {
    # Allowlist marker check (top of file)
    $Allowlisted = $false
    try {
        $HeadContent = git show ":$f" 2>$null | Select-Object -First 30
        if ($HeadContent -match 'duckbot-secret-scan:\s*allowlist-file') {
            $Allowlisted = $true
        }
    } catch {
        continue
    }
    if ($Allowlisted) {
        continue
    }

    # Get the staged content
    $StagedContent = $null
    try {
        $StagedContent = git show ":$f" 2>$null
    } catch {
        continue
    }
    if (-not $StagedContent) {
        continue
    }

    foreach ($Pattern in $SecretPatterns) {
        $Label = $Pattern.Label
        $Regex = $Pattern.Regex
        $Hits = @()
        # Line-by-line scan with line numbers
        $LineNum = 0
        foreach ($Line in $StagedContent) {
            $LineNum++
            if ($Line -match $Regex) {
                $Hits += "$LineNum`: $Line"
                if ($Hits.Count -ge 5) { break }
            }
        }
        if ($Hits.Count -gt 0) {
            Write-Host "❌ Secret pattern detected: $Label in $f" -ForegroundColor Red
            $Hits | ForEach-Object { Write-Host "    $_" }
            $Found = $true
        }
    }
}

# --- 5. Forbidden paths -----------------------------------------------------

$ForbiddenPatterns = @(
    '^\.env$',
    '^\.env\.local$',
    '^\.env\.prod(uction)?$',
    '^\.env\.staging$',
    '^data/chroma/',
    '^data/watcher_state\.json$',
    '^data/ingest_history\.jsonl$',
    '^data/eval_history\.jsonl$',
    '^\.venv/',
    '^node_modules/'
)

foreach ($PathRe in $ForbiddenPatterns) {
    foreach ($f in $StagedFiles) {
        if ($f -match $PathRe) {
            Write-Host "❌ Forbidden path staged: $f" -ForegroundColor Red
            $Found = $true
        }
    }
}

# --- 6. Report ---------------------------------------------------------------

if ($Found) {
    Write-Host ""
    Write-Host "🛑 COMMIT BLOCKED. One or more secrets or forbidden paths were detected." -ForegroundColor Red
    Write-Host ""
    Write-Host "How to fix:"
    Write-Host "  1. If this is a real leak: ROTATE THE CREDENTIAL IMMEDIATELY before retrying."
    Write-Host "  2. If it's a false positive (test fixture, example, etc.):"
    Write-Host "     - Use a placeholder like 'sk-EXAMPLE-NOT-A-REAL-KEY' (short, won't match)."
    Write-Host "     - Add the file to .gitignore if it shouldn't be tracked at all."
    Write-Host "  3. If you really must override (emergency hotfix only):"
    Write-Host "     `$env:DUCKBOT_SKIP_SECRET_SCAN=1; git commit --no-verify"
    Write-Host "     ⚠️ This is logged. Don't do it unless you're sure."
    Write-Host ""
    Write-Host "If you need help, ask the operator."
    exit 1
}

exit 0
