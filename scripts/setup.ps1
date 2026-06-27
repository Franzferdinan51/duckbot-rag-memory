# setup.ps1 — DuckBot one-click setup for Windows (PowerShell).
#
# This is the recommended way to set up DuckBot on Windows.
# Double-click setup.bat in Explorer, or run:
#   pwsh .\scripts\setup.ps1
#
# It runs the full install (venv, deps, .env, Task Scheduler) then
# seeds the demo corpus and runs a sample query.

# Resolve repo root
$RepoRoot = $null
try {
    $gitRoot = git rev-parse --show-toplevel 2>$null
    if ($gitRoot) { $RepoRoot = $gitRoot }
} catch {}
if (-not $RepoRoot) {
    $current = (Resolve-Path .).Path
    while ($true) {
        if (Test-Path (Join-Path $current "src\watcher.py")) { $RepoRoot = $current; break }
        $parent = Split-Path $current -Parent
        if ($parent -eq $current) { break }
        $current = $parent
    }
}
if (-not $RepoRoot) {
    Write-Error "Could not find repo root. Run from the duckbot-rag-memory directory."
    exit 1
}
Set-Location $RepoRoot

Write-Host ""
Write-Host "🧠  DuckBot RAG + Memory — One-Click Setup (Windows)"
Write-Host "    Repo: $RepoRoot"
Write-Host ""

# Delegate to install.ps1 (full install + Task Scheduler + demo)
& "$RepoRoot\scripts\install.ps1" -SkipVenv:$false -SkipTask:$false
