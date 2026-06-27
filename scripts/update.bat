@echo off
:: update.bat — Update DuckBot to the latest version (Windows).
:: Double-click this file, or run from Command Prompt / PowerShell.
::
:: For agents / machine use (JSON output):
::   .\venv\Scripts\python.exe -m src.cli update
::   .\venv\Scripts\python.exe -m src.cli update --dry-run

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

:: Run the update
echo  Calling: python -m src.cli update
echo.
"%PYTHON%" -m src.cli update %*
set UPDATE_RESULT=%ERRORLEVEL%
echo.

if %UPDATE_RESULT% neq 0 (
    echo  ⚠ Update returned an error code: %UPDATE_RESULT%
) else (
    echo  ✅ Update complete.
)
echo.
echo For agent use (JSON output):
echo   %PYTHON% -m src.cli update
echo   %PYTHON% -m src.cli update --dry-run
echo.
pause
