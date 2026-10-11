@echo off
rem ==============================================================================
rem VoxFlow ML Engine — 1-Click Installer (Windows)
rem ==============================================================================
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ======================================================
echo          VoxFlow ML Engine - Windows Setup
echo ======================================================
echo.

rem 1. Check Python
where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PY_CMD=python"
    goto :PYTHON_FOUND
)

where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set "PY_CMD=py -3"
    goto :PYTHON_FOUND
)

echo [ERROR] Python 3.10+ was not found on your system!
echo Please download and install Python from: https://www.python.org/downloads/
echo Note: Make sure to check "Add Python to PATH" during installation.
pause
exit /b 1

:PYTHON_FOUND
echo [*] Found Python command: %PY_CMD%
%PY_CMD% --version

rem Check version >= 3.10
%PY_CMD% -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python 3.10 or higher is required. Please upgrade your Python version.
    pause
    exit /b 1
)

rem 2. Create Virtual Environment
if not exist ".venv\Scripts\activate.bat" (
    echo [*] Creating virtual environment (.venv)...
    %PY_CMD% -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment already exists.
)

rem 3. Activate Virtual Environment
call .venv\Scripts\activate.bat

rem 4. Upgrade pip and install requirements
echo [*] Updating pip and installing dependencies...
python -m pip install --quiet --upgrade pip setuptools wheel
if exist "requirements.txt" (
    python -m pip install -r requirements.txt
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to install requirements.
        pause
        exit /b 1
    )
    echo [OK] All dependencies successfully installed.
) else (
    echo [WARN] requirements.txt not found.
)

rem 5. Quick Sanity Verification
echo [*] Verifying engine components...
python -c "import numpy; import onnxruntime; from vad_service import vad_service; print('[OK] Core imports and VAD Service verified.')"

echo.
echo ======================================================
echo      VoxFlow Windows Setup Completed Successfully!
echo ======================================================
echo.
echo You can now launch VoxFlow with:
echo   run.bat              - Start Web Studio and WebSocket server
echo   run.bat test         - Launch Interactive Voice Tester
echo   run.bat benchmark    - Run Latency Evaluation Benchmark
echo   run.bat pytest       - Run Automated Test Suite
echo.
pause
