@echo off
:: setup.bat — Launch the DuckBot one-click setup on Windows.
:: Double-click this file, or run from Command Prompt / PowerShell.

echo.
echo 🧠  DuckBot RAG + Memory — One-Click Setup (Windows)
echo.

:: Detect PowerShell
where pwsh >nul 2>&1
if %errorlevel% equ 0 (
    echo  Detected PowerShell — launching setup...
    echo.
    pwsh -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
) else (
    echo  ⚠ PowerShell (pwsh) not found.
    echo.
    echo  Install PowerShell from: https://github.com/PowerShell/PowerShell/releases
    echo  Or install Python directly and run:
    echo    python -m venv .venv
    echo    .venv\Scripts\pip install -r requirements.txt
    echo    copy .env.example .env
    echo    .venv\Scripts\python -m src.cli doctor
    echo.
    pause
)
