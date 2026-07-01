@echo off
rem duck-memory-mcp.bat — OpenClaw brain MCP launcher (separate data dir)
rem Sets DUCK_MEM_DIR before launching DuckMemory MCP server.
rem Prevents Hermès from sharing OpenClaw's memories.
setlocal enabledelayedexpansion

rem Resolve repo root (script is in scripts/ subdir)
set "REPO=%~dp0.."
set "REPO=%REPO:~0,-1%"

rem Activate venv
set "VENV=%REPO%\.venv\Scripts\python.exe"
if not exist "%VENV%" set "VENV=%REPO%\.venv\bin\python.exe"

rem OpenClaw brain data (separate from Hermès)
set "DUCK_MEM_DIR=%USERPROFILE%\.duck-memory"

rem Set DuckMemory data path so both brains can run simultaneously
set "DUCKBOT_DATA_DIR=%DUCK_MEM_DIR%\brain"

rem Embedding + LM Studio
set "PYTHONPATH=%REPO%"
set "PYTHONIOENCODING=utf-8"

rem Delegate to MCP server
"%VENV%" -m src.mcp_server %*
