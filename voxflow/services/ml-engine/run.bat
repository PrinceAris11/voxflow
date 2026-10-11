@echo off
rem ==============================================================================
rem VoxFlow ML Engine — 1-Click Launcher (Windows)
rem ==============================================================================
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo [INFO] Virtual environment not found. Running 1-click installer first...
    call install.bat
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Installation failed.
        pause
        exit /b 1
    )
)

call .venv\Scripts\activate.bat

set "MODE=%~1"

if "%MODE%"=="" goto :START_SERVER
if /i "%MODE%"=="server" goto :START_SERVER
if /i "%MODE%"=="studio" goto :START_SERVER
if /i "%MODE%"=="test" goto :RUN_TEST
if /i "%MODE%"=="interactive" goto :RUN_TEST
if /i "%MODE%"=="benchmark" goto :RUN_BENCHMARK
if /i "%MODE%"=="bench" goto :RUN_BENCHMARK
if /i "%MODE%"=="pytest" goto :RUN_PYTEST
if /i "%MODE%"=="tests" goto :RUN_PYTEST

rem Custom command
python %*
goto :EOF

:START_SERVER
echo ======================================================
echo   Launching VoxFlow Web Studio and WebSocket Server
echo ======================================================
echo    * Web Studio UI:     http://127.0.0.1:8000
echo    * Floating Pill:     http://127.0.0.1:8000/widget
echo    * WebSocket Stream:  ws://127.0.0.1:8000/ws
echo    * Health Check:      http://127.0.0.1:8000/health
echo.
echo Press Ctrl+C in this window to stop the server.
echo.
python main.py
goto :EOF

:RUN_TEST
echo Launching Interactive Voice and VAD Tester...
python interactive_test.py
goto :EOF

:RUN_BENCHMARK
echo Running Silero VAD and Pipeline Latency Benchmarks...
python vad_benchmark.py
goto :EOF

:RUN_PYTEST
echo Running automated pytest test suite...
pytest tests -v
goto :EOF
