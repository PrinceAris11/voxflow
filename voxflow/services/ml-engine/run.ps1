<#
.SYNOPSIS
    VoxFlow ML Engine — 1-Click Launcher (PowerShell)
#>
param (
    [string]$Mode = "server"
)

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$VenvDir = Join-Path $ScriptDir ".venv"
$VenvPython = if ($IsWindows -or ($env:OS -like "*Windows*")) {
    Join-Path $VenvDir "Scripts\python.exe"
} else {
    Join-Path $VenvDir "bin/python"
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "Virtual environment not detected. Running installer first..." -ForegroundColor Yellow
    & (Join-Path $ScriptDir "install.ps1")
}

switch ($Mode.ToLower()) {
    { $_ -in "server", "studio", "app", "" } {
        Write-Host "======================================================" -ForegroundColor Cyan
        Write-Host "   🚀 Launching VoxFlow Web Studio & WebSocket Server  " -ForegroundColor Cyan
        Write-Host "======================================================" -ForegroundColor Cyan
        Write-Host "   * Studio UI:     http://127.0.0.1:8000"
        Write-Host "   * Floating Pill: http://127.0.0.1:8000/widget"
        Write-Host "   * WebSocket:     ws://127.0.0.1:8000/ws"
        Write-Host "   * Health Check:  http://127.0.0.1:8000/health"
        Write-Host ""
        Write-Host "Press Ctrl+C to stop the server." -ForegroundColor Yellow
        & $VenvPython "main.py"
    }
    { $_ -in "test", "interactive" } {
        Write-Host "🎙️ Launching Interactive Voice & VAD Tester..." -ForegroundColor Cyan
        & $VenvPython "interactive_test.py"
    }
    { $_ -in "benchmark", "bench" } {
        Write-Host "⚡ Running Silero VAD & Pipeline Latency Benchmark..." -ForegroundColor Cyan
        & $VenvPython "vad_benchmark.py"
    }
    { $_ -in "pytest", "tests" } {
        Write-Host "🧪 Running automated pytest test suite..." -ForegroundColor Cyan
        & $VenvPython -m pytest tests -v
    }
    default {
        & $VenvPython -c "import sys; print('Running custom mode:', sys.argv)"
    }
}
