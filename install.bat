@echo off
rem ==============================================================================
rem VoxFlow Project Root — 1-Click Installer (Windows)
rem ==============================================================================
setlocal
cd /d "%~dp0"
if exist "voxflow\services\ml-engine\install.bat" (
    cd "voxflow\services\ml-engine"
    call install.bat %*
) else (
    echo [ERROR] ML Engine directory not found at voxflow\services\ml-engine
    pause
)
