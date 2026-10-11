#!/usr/bin/env bash
# ==============================================================================
# VoxFlow ML Engine — 1-Click Runner (Linux & macOS)
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BOLD='\033[1m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Auto-install if virtual environment does not exist
if [ ! -d "$SCRIPT_DIR/.venv" ] || [ ! -f "$SCRIPT_DIR/.venv/bin/activate" ]; then
    echo -e "${YELLOW}ℹ Virtual environment not detected. Running 1-click installation first...${NC}"
    bash "$SCRIPT_DIR/install.sh"
fi

# Activate virtual environment
# shellcheck source=/dev/null
source "$SCRIPT_DIR/.venv/bin/activate"

# Handle arguments
MODE="${1:-server}"

case "$MODE" in
    server|studio|app|"")
        echo -e "${CYAN}${BOLD}🚀 Launching VoxFlow Web Studio & WebSocket Server...${NC}"
        echo -e "   → Studio UI:     ${BOLD}http://127.0.0.1:8000${NC}"
        echo -e "   → Floating Pill: ${BOLD}http://127.0.0.1:8000/widget${NC}"
        echo -e "   → WebSocket:     ${BOLD}ws://127.0.0.1:8000/ws${NC}"
        echo -e "   → Health Check:  ${BOLD}http://127.0.0.1:8000/health${NC}"
        echo -e "${YELLOW}Press Ctrl+C to stop the server.${NC}"
        echo ""
        exec python main.py
        ;;
    test|interactive)
        echo -e "${CYAN}${BOLD}🎙️ Launching VoxFlow Interactive Audio & VAD Tester...${NC}"
        exec python interactive_test.py
        ;;
    benchmark|bench)
        echo -e "${CYAN}${BOLD}⚡ Running Silero VAD & Pipeline Latency Benchmarks...${NC}"
        exec python vad_benchmark.py "${@:2}"
        ;;
    pytest|test-suite|tests)
        echo -e "${CYAN}${BOLD}🧪 Running automated pytest test suite...${NC}"
        exec pytest tests -v "${@:2}"
        ;;
    help|--help|-h)
        echo -e "${BOLD}VoxFlow ML Engine Runner${NC}"
        echo -e "Usage:"
        echo -e "  ${GREEN}./run.sh${NC}              Start Web Studio & WebSocket server (default)"
        echo -e "  ${GREEN}./run.sh test${NC}         Launch Interactive Voice & VAD Tester CLI"
        echo -e "  ${GREEN}./run.sh benchmark${NC}    Run latency evaluation benchmark suite"
        echo -e "  ${GREEN}./run.sh pytest${NC}       Run automated pytest tests"
        echo -e "  ${GREEN}./run.sh <command>${NC}    Execute any python command within the environment"
        ;;
    *)
        # Pass custom commands directly to python or shell in venv
        exec "$@"
        ;;
esac
