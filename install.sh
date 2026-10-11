#!/usr/bin/env bash
# ==============================================================================
# VoxFlow Project Root — 1-Click Installer (Linux & macOS)
# ==============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$ROOT_DIR/voxflow/services/ml-engine"

if [ -f "$ENGINE_DIR/install.sh" ]; then
    bash "$ENGINE_DIR/install.sh" "$@"
else
    echo "Error: ML Engine directory not found at $ENGINE_DIR"
    exit 1
fi
