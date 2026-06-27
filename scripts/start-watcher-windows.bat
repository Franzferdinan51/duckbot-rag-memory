@echo off
REM Start the watcher fully detached on Windows.
REM Run via: cmd //c scripts\start-watcher-windows.bat
setlocal
cd /d "%~dp0\.."
del /q data\watcher.pid 2>nul
REM Start pythonw so no console window pops up. Use python.exe if you want stdout.
start "" /B .\venv\Scripts\pythonw.exe -m src.watcher run
echo Started watcher.
endlocal
