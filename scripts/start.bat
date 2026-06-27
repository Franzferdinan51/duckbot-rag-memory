@echo off
:: start.bat — Start the DuckBot memory watcher daemon (Windows).
:: Double-click, or run from Command Prompt / PowerShell.

echo.
echo 🧠  DuckBot — Starting watcher daemon
echo.

:: Find Python
if exist "%~dp0..\.venv\Scripts\python.exe" (
    set PYTHON=%~dp0..\.venv\Scripts\python.exe
) else if exist "%~dp0..\.venv\Scripts\pythonw.exe" (
    set PYTHON=%~dp0..\.venv\Scripts\pythonw.exe
) else (
    echo  ⚠ No venv found. Run .\scripts\setup.bat first.
    pause
    exit /b 1
)

:: Check if already running
for /f "delims=" %%s in ('"%PYTHON%" -m src.watcher status 2^>nul') do set "STATUS=%%s"
if defined STATUS (
    echo %STATUS%
    if not "%STATUS:running=%"=="%STATUS%" (
        echo   Watcher is already running.
        pause
        exit /b 0
    )
)

:: Load .env
if exist "%~dp0..\.env" (
    for /f "usebackq tokens=1,* delims==" %%A in (`findstr /v "#" "%~dp0..\.env" ^| find "="`) do set "%%A=%%B"
)

:: Build watch paths
set "WATCH_ARGS="
if exist "%OPENCLAW_MEMORY%" (
    set "WATCH_ARGS=%OPENCLAW_MEMORY%"
)
:: Add repo docs
for %%f in (AGENTS.md SOUL.md USER.md IDENTITY.md TOOLS.md README.md) do (
    if exist "%~dp0..\%%f" (
        if defined WATCH_ARGS (
            set "WATCH_ARGS=!WATCH_ARGS! %~dp0..\%%f"
        ) else (
            set "WATCH_ARGS=%~dp0..\%%f"
        )
    )
)

:: Create data dir
if not exist "%~dp0..\data" mkdir "%~dp0..\data"

:: Start watcher
echo  Starting watcher (polls every 5 min, content-hash dedup)...
echo  Log: %~dp0..\data\watcher.log
start /b "" "%PYTHON%" -m src.watcher run %WATCH_ARGS% >>"%~dp0..\data\watcher.log" 2>&1

timeout /t 3 /nobreak >nul
for /f "delims=" %%s in ('"%PYTHON%" -m src.watcher status 2^>nul') do set "STATUS=%%s"
if defined STATUS (
    echo.
    echo  Status: %STATUS%
) else (
    echo.
    echo  ⚠ Could not confirm watcher started. Check:
    echo    type %~dp0..\data\watcher.log
)

echo.
echo Manage:
echo   "%PYTHON%" -m src.watcher status
echo   "%PYTHON%" -m src.watcher stop
echo   type %~dp0..\data\watcher.log
echo.
pause
