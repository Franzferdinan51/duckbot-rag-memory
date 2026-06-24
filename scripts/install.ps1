# install.ps1 — bootstrap the DuckBot brain on Windows.
#
# duckbot-secret-scan: allowlist-file
#
# Cross-platform companion to scripts/install-macos.sh (launchd) and
# scripts/install-linux.sh (systemd). On Windows, the equivalent of
# "auto-restart on crash + on boot" is **Task Scheduler**.
#
# What it does:
#   1. Creates .venv (if missing) via `python -m venv`
#   2. Installs deps from requirements.txt
#   3. Copies .env.example → .env (if .env missing)
#   4. Registers a Task Scheduler task that runs the watcher on logon
#
# Usage (from repo root in PowerShell, as a regular user):
#   pwsh scripts/install.ps1
#
# Manage with:
#   Get-ScheduledTask -TaskName "DuckBotMemoryWatcher"
#   Start-ScheduledTask -TaskName "DuckBotMemoryWatcher"
#   Stop-ScheduledTask -TaskName "DuckBotMemoryWatcher"
#   Unregister-ScheduledTask -TaskName "DuckBotMemoryWatcher" -Confirm:$false

[CmdletBinding()]
param(
    [switch]$SkipVenv,
    [switch]$SkipTask,
    [switch]$Unregister
)

$ErrorActionPreference = 'Stop'

# --- 1. Resolve repo root + paths -------------------------------------------

# Resolve the repo root, with fallbacks for Windows machines that don't
# have `git` on PATH (rare, but happens on locked-down corporate boxes).
# Walk up from $PSScriptRoot until we find a directory containing
# `src/watcher.py` — that's the unambiguous repo marker.
function Resolve-RepoRoot {
    param([string]$Start)
    $current = (Resolve-Path $Start).Path
    while ($true) {
        if (Test-Path (Join-Path $current "src\watcher.py")) {
            return $current
        }
        $parent = Split-Path $current -Parent
        if ($parent -eq $current) {
            return $null  # reached filesystem root without finding the marker
        }
        $current = $parent
    }
}

$RepoRoot = $null
try {
    $gitRoot = git rev-parse --show-toplevel 2>$null
    if ($gitRoot) { $RepoRoot = $gitRoot }
} catch {}

if (-not $RepoRoot) {
    $RepoRoot = Resolve-RepoRoot -Start $PSScriptRoot
}
if (-not $RepoRoot) {
    Write-Error "Could not locate the repo root. cd to the repo root and re-run."
    exit 1
}
Set-Location $RepoRoot

Write-Host "=== DuckBot RAG + Memory install (Windows) ==="
Write-Host "Repo: $RepoRoot"
Write-Host "OS:   $($PSVersionTable.OS)"

# --- 2. Unregister mode -----------------------------------------------------

if ($Unregister) {
    if (Get-ScheduledTask -TaskName "DuckBotMemoryWatcher" -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName "DuckBotMemoryWatcher" -Confirm:$false
        Write-Host "✓ Unregistered scheduled task 'DuckBotMemoryWatcher'"
    } else {
        Write-Host "No scheduled task named 'DuckBotMemoryWatcher' to unregister."
    }
    exit 0
}

# --- 3. Resolve python executable ------------------------------------------

$PythonExe = $null
$Candidates = @(
    (Join-Path $RepoRoot ".venv\Scripts\python.exe"),
    (Join-Path $RepoRoot ".venv\Scripts\pythonw.exe"),
    (Get-Command python -ErrorAction SilentlyContinue).Path,
    (Get-Command py -ErrorAction SilentlyContinue).Path
)
foreach ($c in $Candidates) {
    if ($c -and (Test-Path $c)) { $PythonExe = $c; break }
}
if (-not $PythonExe) {
    Write-Error "No Python found. Install Python 3.9+ and re-run."
    exit 1
}
Write-Host "Using Python: $PythonExe"

# --- 4. venv + deps --------------------------------------------------------

$VenvDir = Join-Path $RepoRoot ".venv"
if (-not $SkipVenv) {
    if (-not (Test-Path $VenvDir)) {
        Write-Host "Creating venv..."
        & $PythonExe -m venv $VenvDir
    } else {
        Write-Host "Venv already exists at $VenvDir"
    }
    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"
    $VenvPip = Join-Path $VenvDir "Scripts\pip.exe"

    Write-Host "Upgrading pip..."
    & $VenvPython -m pip install --quiet --upgrade pip

    $ReqFile = Join-Path $RepoRoot "requirements.txt"
    if (Test-Path $ReqFile) {
        Write-Host "Installing deps from requirements.txt..."
        & $VenvPip install --quiet -r $ReqFile
    } else {
        Write-Warning "No requirements.txt; skipping pip install"
    }
} else {
    Write-Host "Skipping venv setup (--SkipVenv)"
}

# --- 5. .env ---------------------------------------------------------------

$EnvFile = Join-Path $RepoRoot ".env"
$EnvExample = Join-Path $RepoRoot ".env.example"
if ((-not (Test-Path $EnvFile)) -and (Test-Path $EnvExample)) {
    Copy-Item $EnvExample $EnvFile
    Write-Host "Created .env from template. EDIT IT to set LMSTUDIO_URL, LMSTUDIO_KEY, MINIMAX_API_KEY."
} elseif (Test-Path $EnvFile) {
    Write-Host ".env already exists"
} else {
    Write-Warning "No .env or .env.example; skipping"
}

# --- 6. Task Scheduler ----------------------------------------------------

if (-not $SkipTask) {
    $TaskName = "DuckBotMemoryWatcher"
    $TaskDescription = "DuckBot memory watcher — auto-updates the brain from filesystem changes"

    # Build the watcher command. Use pythonw.exe for no console window.
    $WatchPython = Join-Path $VenvDir "Scripts\pythonw.exe"
    if (-not (Test-Path $WatchPython)) { $WatchPython = Join-Path $VenvDir "Scripts\python.exe" }
    $WatchPaths = @(
        (Join-Path $RepoRoot "memory"),
        (Join-Path $RepoRoot "AGENTS.md"),
        (Join-Path $RepoRoot "SOUL.md"),
        (Join-Path $RepoRoot "USER.md"),
        (Join-Path $RepoRoot "IDENTITY.md"),
        (Join-Path $RepoRoot "TOOLS.md")
    ) | Where-Object { Test-Path $_ }
    $WatchArgs = @("-m", "src.watcher", "run") + $WatchPaths

    $Action = New-ScheduledTaskAction `
        -Execute $WatchPython `
        -Argument ($WatchArgs -join " ") `
        -WorkingDirectory $RepoRoot

    # Trigger: at logon. Auto-restart on failure.
    $Trigger = New-ScheduledTaskTrigger -AtLogOn

    # Settings: restart on failure with 1-minute delay, allow start on
    # batteries, run only when network available.
    $Settings = New-ScheduledTaskSettingsSet `
        -RestartCount 5 `
        -RestartInterval (New-TimeSpan -Minutes 1) `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable

    # Principal: current user, run with highest privileges so it survives
    # logon/logoff cycles. S4U / Interactive only.
    $Principal = New-ScheduledTaskPrincipal `
        -UserId $env:USERNAME `
        -LogonType Interactive `
        -RunLevel Highest

    # Register (or replace) the task
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Write-Host "Updating existing scheduled task '$TaskName'..."
        Set-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal | Out-Null
    } else {
        Write-Host "Registering scheduled task '$TaskName'..."
        Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings -Principal $Principal -Description $TaskDescription | Out-Null
    }

    # Start it now
    Start-ScheduledTask -TaskName $TaskName
    Write-Host "✓ Scheduled task '$TaskName' registered and started."
} else {
    Write-Host "Skipping task registration (--SkipTask)"
}

# --- 7. Pre-commit hook ----------------------------------------------------

$HookPath = Join-Path $RepoRoot ".git\hooks\pre-commit"
$HookPs1 = Join-Path $RepoRoot "scripts\secret-scan.ps1"
if ((Test-Path $HookPath) -and (Select-String -Path $HookPath -Pattern 'secret-scan' -SimpleMatch -Quiet)) {
    Write-Host "✓ Pre-commit hook already installed at $HookPath"
} else {
    Write-Host ""
    Write-Host "To install the pre-commit secret-scan hook, run:"
    Write-Host "  pwsh scripts/install-pre-commit.ps1"
}

# --- 8. Done ---------------------------------------------------------------

Write-Host ""
Write-Host "=== Install complete ==="
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Edit .env to set LMSTUDIO_URL + LMSTUDIO_KEY (and optional MiniMax key for fallback)"
Write-Host "  2. .\.venv\Scripts\python.exe -m src.cli doctor                    # verify all green"
Write-Host "  3. .\.venv\Scripts\python.exe -m src.watcher once                  # cold-start full sync"
Write-Host "  4. pwsh scripts/start-watcher.ps1                                 # start in background now"
Write-Host "  5. pwsh scripts/start-watcher.ps1 -Status                        # check status"
Write-Host "  6. pwsh scripts/start-watcher.ps1 -Log                           # tail logs"
Write-Host ""
Write-Host "Manage the scheduled task:"
Write-Host "  Get-ScheduledTask -TaskName 'DuckBotMemoryWatcher'"
Write-Host "  Stop-ScheduledTask -TaskName 'DuckBotMemoryWatcher'"
Write-Host "  Unregister-ScheduledTask -TaskName 'DuckBotMemoryWatcher' -Confirm:`$false"
