@echo off
:: duck-memory.bat — One-command entry point to the DuckBot memory CLI (Windows).
::
:: Drop this anywhere on your PATH, or run it from the repo root.
:: Usage:
::   duck-memory                    :: show CLI help
::   duck-memory query "question"
::   duck-memory ingest myfile.md
::   duck-memory --help

:: Find repo root: where this .bat lives
set "REPO_ROOT=%~dp0"
set "REPO_ROOT=%REPO_ROOT:~0,-1%"

:: Find venv python
if exist "%REPO_ROOT%\.venv\Scripts\python.exe" (
    set PYTHON=%REPO_ROOT%\.venv\Scripts\python.exe
) else if exist "%REPO_ROOT%\.venv\Scripts\pythonw.exe" (
    set PYTHON=%REPO_ROOT%\.venv\Scripts\pythonw.exe
) else (
    echo duck-memory: no venv found at %REPO_ROOT%\.venv
    echo Run .\scripts\setup.bat first to set up the environment.
    exit /b 1
)

:: Load .env (basic key=value, skips comments and blank lines)
if exist "%REPO_ROOT%\.env" (
    for /f "usebackq tokens=1,* delims==" %%A in (`findstr /v "#" "%REPO_ROOT%\.env" ^| find "="`) do set "%%A=%%B"
)

:: Delegate to the CLI
"%PYTHON%" -m src.cli %*
