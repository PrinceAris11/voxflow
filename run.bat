@echo off
rem ==============================================================================
rem VoxFlow Project Root — 1-Click Launcher (Windows)
rem ==============================================================================
setlocal
cd /d "%~dp0"
if exist "voxflow\services\ml-engine\run.bat" (
    cd "voxflow\services\ml-engine"
    call run.bat %*
) else (
    echo [ERROR] ML Engine directory not found at voxflow\services\ml-engine
    pause
)
