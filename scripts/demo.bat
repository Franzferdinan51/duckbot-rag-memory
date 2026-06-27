@echo off
:: demo.bat — Run the DuckBot demo (Windows).
:: Double-click this file, or run from Command Prompt / PowerShell.
:: For PowerShell, use: .\scripts\demo.bat

echo.
echo 🧠  DuckBot RAG + Memory — Demo
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

:: Load .env
if exist "%~dp0..\.env" (
    for /f "usebackq tokens=1,* delims==" %%A in (`findstr /v "#" "%~dp0..\.env" ^| find "="`) do set "%%A=%%B"
)

:: Step 1: Doctor
echo [1/4] Verifying setup...
"%PYTHON%" -m src.cli doctor 2>nul | findstr /i "error failed" >nul
if errorlevel 1 (
    echo   ✓ All checks passed
)
echo.

:: Step 2: Seed
echo [2/4] Seeding demo corpus (idempotent)...
"%PYTHON%" -m src.cli seed-demo 2>nul
echo.

:: Step 3: Query BATMAN
echo [3/4] Querying: How do I restart the BATMAN container?
"%PYTHON%" -m src.cli query "How do I restart the BATMAN container?" -n 3 2>nul
echo.

:: Step 4: Query design
echo [4/4] Querying: What are DuckBot's design constraints?
"%PYTHON%" -m src.cli query "What are DuckBot's design constraints?" -n 3 2>nul
echo.

echo ✅ Demo complete.
echo.
echo Next:
echo   .\scripts\duckbot-ask "your question"
echo   .\scripts\start.bat       # start watcher daemon
echo   .\scripts\openclaw-bootstrap.bat   # set up with OpenClaw
echo   .\scripts\hermes-bootstrap.bat    # set up with Hermes Agent
echo.
pause
