@echo off
REM Hermes MCP launcher for duckbot-brain (Windows native).
REM Cross-platform companion to scripts/duckbot-memory-mcp.sh.
REM
REM Loads the brain's .env (LMSTUDIO_KEY, OPENAI_API_KEY, MINIMAX_API_KEY, etc.)
REM and starts the stdio MCP server. Hermes reads this script's stdout line-by-line.
REM
REM Hermes on Windows expects a real .exe/.bat for stdio commands, not a .sh
REM script. This wrapper is the same as the .sh version, but cmd.exe-native.
REM
REM Install into Hermes:
REM   hermes mcp add duckbot-memory ^
REM     --env "PYTHONUNBUFFERED=1" ^
REM     --command "C:\Users\franz\Desktop\duckbot-rag-memory\scripts\duckbot-memory-mcp.bat"
REM
REM (Or pass --args to override defaults.)

setlocal
set "REPO_ROOT=%~dp0.."
cd /d "%REPO_ROOT%"

REM Load .env if present. The previous `for /f ... delims==` parsed
REM incorrectly: values containing `=`, `&`, `|`, `<`, `>`, `%` were
REM mangled, comments/blank lines weren't skipped, and trailing `=`
REM was lost. Use findstr to skip blanks/comments first, then split
REM ONLY on the first `=`.
if exist "%REPO_ROOT%\.env" (
    for /f "usebackq tokens=1* delims==" %%A in (`
        findstr /v /r /c:"^$" /c:"^#;" "%REPO_ROOT%\.env"
    `) do (
        if not "%%A"=="" set "%%A=%%B"
    )
)

REM Hermes reads MCP stdio line-by-line; ensure unbuffered output.
set "PYTHONUNBUFFERED=1"

REM Detect venv python (Windows venvs live in Scripts\, POSIX in bin\)
if exist "%REPO_ROOT%\.venv\Scripts\python.exe" (
    set "PYTHON_BIN=%REPO_ROOT%\.venv\Scripts\python.exe"
) else if exist "%REPO_ROOT%\.venv\bin\python.exe" (
    set "PYTHON_BIN=%REPO_ROOT%\.venv\bin\python.exe"
) else (
    echo ❌ No venv python found at %REPO_ROOT%\.venv\Scripts\python.exe 1>&2
    exit /b 1
)

"%PYTHON_BIN%" -m src.mcp_server %*
