<#
.SYNOPSIS
    VoxFlow ML Engine — 1-Click Installer (PowerShell for Windows & macOS/Linux)
#>
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "         🎙️ VoxFlow ML Engine — 1-Click Setup         " -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan

# 1. Detect Python
$PythonCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    $found = Get-Command $cmd -ErrorAction SilentlyContinue
    if ($found) {
        $verCheck = & $cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($verCheck -and [version]$verCheck -ge [version]"3.10") {
            $PythonCmd = $cmd
            Write-Host "✓ Found compatible Python: $cmd ($verCheck)" -ForegroundColor Green
            break
        }
    }
}

if (-not $PythonCmd) {
    Write-Host "✗ Error: Python 3.10 or higher is required." -ForegroundColor Red
    Write-Host "Please install Python from https://www.python.org/downloads/ and ensure it is added to your PATH." -ForegroundColor Yellow
    exit 1
}

# 2. Check virtual environment
$VenvDir = Join-Path $ScriptDir ".venv"
$VenvPython = if ($IsWindows -or ($env:OS -like "*Windows*")) {
    Join-Path $VenvDir "Scripts\python.exe"
} else {
    Join-Path $VenvDir "bin/python"
}

if (-not (Test-Path $VenvPython)) {
    Write-Host "→ Creating virtual environment at .venv..." -ForegroundColor Cyan
    & $PythonCmd -m venv $VenvDir
    Write-Host "✓ Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "✓ Virtual environment already exists." -ForegroundColor Green
}

# 3. Upgrade pip and install requirements
Write-Host "→ Installing dependencies from requirements.txt..." -ForegroundColor Cyan
& $VenvPython -m pip install --quiet --upgrade pip setuptools wheel
if (Test-Path (Join-Path $ScriptDir "requirements.txt")) {
    & $VenvPython -m pip install -r (Join-Path $ScriptDir "requirements.txt")
    Write-Host "✓ Dependencies installed successfully." -ForegroundColor Green
}

# 4. Verification
Write-Host "→ Verifying engine integrity..." -ForegroundColor Cyan
& $VenvPython -c "import numpy; import onnxruntime; from vad_service import vad_service; print('✓ Core modules verified.')"

Write-Host ""
Write-Host "======================================================" -ForegroundColor Green
Write-Host "         🎉 VoxFlow Setup Completed Successfully!       " -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green
Write-Host "You can now run:"
Write-Host "  .\run.ps1             - Start Web Studio and WebSocket Server"
Write-Host "  .\run.ps1 test        - Run Interactive Voice Tester"
Write-Host "  .\run.ps1 benchmark   - Run VAD Latency Benchmark"
Write-Host "  .\run.ps1 pytest      - Run Automated Test Suite"
Write-Host ""
