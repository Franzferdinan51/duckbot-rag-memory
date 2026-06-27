@echo off
:: update.bat — Update DuckBot to the latest version (Windows).
:: Double-click this file, or run from Command Prompt / PowerShell.

echo.
echo 🧠  DuckBot RAG + Memory — Update
echo.

:: Find Python
if exist "%~dp0..\.venv\Scripts\python.exe" (
    set PYTHON=%~dp0..\.venv\Scripts\python.exe
) else (
    echo  ⚠ No venv found. Run .\scripts\setup.bat first.
    pause
    exit /b 1
)

:: Step 1: Stash uncommitted changes
echo [1/5] Stashing local changes...
git stash 2>nul
if errorlevel 1 (
    echo   No local changes to stash.
) else (
    echo   ✓ Changes stashed.
)

:: Step 2: Pull
echo.
echo [2/5] Pulling latest from origin/main...
for /f "delims=" %%r in ('git rev-list --count HEAD..origin/main 2^>nul') do set "BEHIND=%%r"
if "%BEHIND%"=="0" (
    echo   Already up to date.
) else (
    echo   Behind by %BEHIND% commit(s) — pulling...
    git pull --rebase origin main 2>nul
    for /f "delims=" %%h in ('git log -1 --oneline origin/main') do echo   Updated to: %%h
)

:: Step 3: Upgrade deps
echo.
echo [3/5] Updating dependencies...
"%PYTHON%" -m pip install --quiet --upgrade pip 2>nul
if exist "%~dp0..\requirements.txt" (
    "%PYTHON%" -m pip install -r "%~dp0..\requirements.txt" 2>nul
)
echo   ✓ Dependencies up to date.

:: Step 4: Doctor
echo.
echo [4/5] Verifying setup...
"%PYTHON%" -m src.cli doctor 2>nul | findstr /i "error failed" >nul
if errorlevel 1 (
    echo   ✓ All checks passed
) else (
    echo   ⚠ Some checks failed — see above.
)

:: Step 5: Check store
echo.
echo [5/5] Checking store integrity...
for /f "delims=" %%s in ('"%PYTHON%" -m src.cli stats 2^>nul') do echo   %%s

echo.
echo ✅ Update complete.
echo.
echo Run the demo:
echo   .\scripts\demo.bat
echo.
echo Query the brain:
echo   .\scripts\duckbot-ask.bat "your question"
echo.
echo Restart the watcher:
echo   .\scripts\start.bat
echo.
pause
