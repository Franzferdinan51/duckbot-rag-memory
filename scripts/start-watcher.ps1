# start-watcher.ps1 — start the DuckBot memory watcher on Windows.
#
# duckbot-secret-scan: allowlist-file
#
# Cross-platform companion to scripts/start-watcher.sh.
# Uses `pythonw.exe` (no console window) + `subprocess` with
# DETACHED_PROCESS for true background execution.
#
# Usage (from repo root in PowerShell):
#   pwsh scripts/start-watcher.ps1                 # start in background
#   pwsh scripts/start-watcher.ps1 -Foreground     # run in current console (Ctrl+C to stop)
#   pwsh scripts/start-watcher.ps1 -Status         # check if running
#   pwsh scripts/start-watcher.ps1 -Stop           # stop the watcher
#   pwsh scripts/start-watcher.ps1 -Log            # tail the watcher log
#
# State files (created in repo's data/ dir):
#   data/watcher.pid  — PID of the running watcher process
#   data/watcher.log  — append-only log

[CmdletBinding()]
param(
    [switch]$Foreground,
    [switch]$Status,
    [switch]$Stop,
    [switch]$Log,
    [string]$PythonExe = ""
)

$ErrorActionPreference = 'Stop'

# --- 1. Resolve repo root + paths -------------------------------------------

$RepoRoot = git rev-parse --show-toplevel 2>$null
if (-not $RepoRoot) {
    Write-Error "Not in a git repo. cd to the repo root and re-run."
    exit 1
}
Set-Location $RepoRoot

$DataDir = Join-Path $RepoRoot "data"
$StateDir = $DataDir
if (-not (Test-Path $StateDir)) { New-Item -ItemType Directory -Path $StateDir | Out-Null }
$PidPath = Join-Path $StateDir "watcher.pid"
$LogPath = Join-Path $StateDir "watcher.log"

# --- 2. Resolve python executable ------------------------------------------

if (-not $PythonExe) {
    # Prefer pythonw.exe (no console window) for background mode.
    $Candidates = @(
        (Join-Path $RepoRoot ".venv\Scripts\pythonw.exe"),
        (Join-Path $RepoRoot ".venv\Scripts\python.exe"),
        (Get-Command pythonw -ErrorAction SilentlyContinue).Path,
        (Get-Command python -ErrorAction SilentlyContinue).Path
    )
    foreach ($c in $Candidates) {
        if ($c -and (Test-Path $c)) { $PythonExe = $c; break }
    }
}
if (-not $PythonExe) {
    Write-Error "No Python found. Activate the venv or set -PythonExe."
    exit 1
}

# --- 3. Subcommands --------------------------------------------------------

if ($Status) {
    if (-not (Test-Path $PidPath)) {
        Write-Host "Watcher: not running"
        exit 1
    }
    $pidVal = (Get-Content $PidPath -Raw).Trim()
    if (-not $pidVal -or -not ($pidVal -match '^\d+$')) {
        Write-Host "Watcher: stale pid file (content: $pidVal)"
        Remove-Item $PidPath -ErrorAction SilentlyContinue
        exit 1
    }
    $proc = Get-Process -Id ([int]$pidVal) -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Host "Watcher: running (pid=$pidVal)"
        exit 0
    } else {
        Write-Host "Watcher: stale pid file (pid=$pidVal not alive)"
        Remove-Item $PidPath -ErrorAction SilentlyContinue
        exit 1
    }
}

if ($Stop) {
    if (-not (Test-Path $PidPath)) {
        Write-Host "Watcher: not running"
        exit 0
    }
    $pidVal = (Get-Content $PidPath -Raw).Trim()
    try {
        $proc = Get-Process -Id ([int]$pidVal) -ErrorAction Stop
        Stop-Process -Id $proc.Id -Force
        Write-Host "Watcher: stopped pid=$pidVal"
    } catch {
        Write-Host "Watcher: pid=$pidVal not alive ($($_.Exception.Message))"
    }
    Remove-Item $PidPath -ErrorAction SilentlyContinue
    exit 0
}

if ($Log) {
    if (-not (Test-Path $LogPath)) {
        Write-Host "Watcher: no log file yet ($LogPath)"
        exit 0
    }
    Get-Content $LogPath -Tail 50 -Wait
    exit 0
}

# --- 4. Start (foreground or background) -----------------------------------

if (Test-Path $PidPath) {
    $existing = (Get-Content $PidPath -Raw).Trim()
    if ($existing -match '^\d+$') {
        $proc = Get-Process -Id ([int]$existing) -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "Watcher already running (pid=$existing)"
            exit 1
        }
    }
    Remove-Item $PidPath -ErrorAction SilentlyContinue
}

if ($Foreground) {
    # Foreground: just exec the watcher, let Ctrl+C stop it.
    Write-Host "Starting watcher in foreground (Ctrl+C to stop)..."
    & $PythonExe -m src.watcher run
    exit $LASTEXITCODE
}

# Background: launch detached.
Write-Host "Starting watcher in background..."
$WatchPaths = @(
    (Join-Path $RepoRoot "memory"),
    (Join-Path $RepoRoot "AGENTS.md"),
    (Join-Path $RepoRoot "SOUL.md"),
    (Join-Path $RepoRoot "USER.md"),
    (Join-Path $RepoRoot "IDENTITY.md"),
    (Join-Path $RepoRoot "TOOLS.md")
)
# Only include paths that exist
$WatchPaths = $WatchPaths | Where-Object { Test-Path $_ }

$Args = @("-m", "src.watcher", "run")
foreach ($p in $WatchPaths) { $Args += @($p) }

try {
    $proc = Start-Process `
        -FilePath $PythonExe `
        -ArgumentList $Args `
        -WindowStyle Hidden `
        -RedirectStandardOutput $LogPath `
        -RedirectStandardError (Join-Path $DataDir "watcher.err.log") `
        -PassThru `
        -WorkingDirectory $RepoRoot
    # Give the child a moment to start
    Start-Sleep -Milliseconds 500
    # Verify it's still alive
    if ($proc.HasExited) {
        Write-Error "Watcher exited immediately (code=$($proc.ExitCode)). Check $LogPath."
        exit 1
    }
    Set-Content -Path $PidPath -Value $proc.Id
    Write-Host "Watcher daemonized: pid=$($proc.Id)"
    Write-Host "  Log: $LogPath"
    Write-Host "  Status: pwsh scripts/start-watcher.ps1 -Status"
    Write-Host "  Stop:   pwsh scripts/start-watcher.ps1 -Stop"
    exit 0
} catch {
    Write-Error "Failed to start watcher: $($_.Exception.Message)"
    exit 1
}
