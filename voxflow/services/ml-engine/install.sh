#!/usr/bin/env bash
# ==============================================================================
# VoxFlow ML Engine — 1-Click Installer (Linux & macOS)
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Text styles
BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}${BOLD}======================================================${NC}"
echo -e "${BLUE}${BOLD}         🎙️ VoxFlow ML Engine — 1-Click Setup         ${NC}"
echo -e "${BLUE}${BOLD}======================================================${NC}"

# 1. Detect Python 3.10+
PYTHON_BIN=""
for cmd in python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" >/dev/null 2>&1; then
        PY_VER=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "0.0")
        MAJOR=$(echo "$PY_VER" | cut -d. -f1)
        MINOR=$(echo "$PY_VER" | cut -d. -f2)
        if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; then
            PYTHON_BIN="$cmd"
            echo -e "${GREEN}✓ Found compatible Python: $($cmd --version) ($cmd)${NC}"
            break
        fi
    fi
done

if [ -z "$PYTHON_BIN" ]; then
    echo -e "${RED}✗ Error: Python 3.10 or higher is required.${NC}"
    echo -e "Please install Python 3.10+ using your package manager:"
    echo -e "  - Ubuntu/Debian:  sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
    echo -e "  - macOS (Homebrew): brew install python"
    echo -e "  - Fedora:         sudo dnf install -y python3 python3-pip"
    echo -e "  - Arch Linux:     sudo pacman -S python python-pip"
    exit 1
fi

# 2. Check virtual environment
VENV_DIR="$SCRIPT_DIR/.venv"
if [ ! -d "$VENV_DIR" ] || [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo -e "${BLUE}→ Creating virtual environment at .venv...${NC}"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created.${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists.${NC}"
fi

# 3. Activate virtual environment
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# 4. Upgrade pip and install dependencies
echo -e "${BLUE}→ Checking and installing dependencies...${NC}"
python -m pip install --quiet --upgrade pip setuptools wheel

if [ -f "requirements.txt" ]; then
    python -m pip install --quiet -r requirements.txt
    echo -e "${GREEN}✓ All dependencies successfully installed.${NC}"
else
    echo -e "${YELLOW}! Warning: requirements.txt not found in $SCRIPT_DIR.${NC}"
fi

# 5. Check optional audio tools (non-blocking)
echo -e "${BLUE}→ Checking optional audio utilities...${NC}"
if command -v ffmpeg >/dev/null 2>&1; then
    echo -e "${GREEN}✓ ffmpeg is available (audio transcoding enabled).${NC}"
else
    echo -e "${YELLOW}ℹ Optional: ffmpeg not detected. To enable full audio format conversions:${NC}"
    echo -e "    Ubuntu/Debian: sudo apt install -y ffmpeg"
    echo -e "    macOS: brew install ffmpeg"
fi

# 6. Verify environment sanity
echo -e "${BLUE}→ Verifying ML Engine core components...${NC}"
python -c '
import numpy as np
import onnxruntime as ort
import fastapi
from vad_service import vad_service
print("✓ Core imports verified.")
print(f"✓ ONNX Runtime: {ort.__version__}")
print(f"✓ VAD Engine ready (mode={vad_service.mode}, sample_rate={vad_service.sample_rate}Hz)")
'

echo ""
echo -e "${GREEN}${BOLD}======================================================${NC}"
echo -e "${GREEN}${BOLD}         🎉 VoxFlow Setup Completed Successfully!       ${NC}"
echo -e "${GREEN}${BOLD}======================================================${NC}"
echo -e "You can now run VoxFlow with:"
echo -e "  ${BOLD}./run.sh${NC}             -> Start Web Studio & WebSocket server"
echo -e "  ${BOLD}./run.sh test${NC}        -> Launch Interactive Voice & VAD Tester"
echo -e "  ${BOLD}./run.sh benchmark${NC}   -> Run VAD & Pipeline Latency Benchmark"
echo -e "  ${BOLD}./run.sh pytest${NC}      -> Run complete automated test suite"
echo ""
