#!/usr/bin/env bash
# ==============================================================================
# VoxFlow Project Root — 1-Click Launcher (Linux & macOS)
# ==============================================================================
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENGINE_DIR="$ROOT_DIR/voxflow/services/ml-engine"

if [ -f "$ENGINE_DIR/run.sh" ]; then
    exec bash "$ENGINE_DIR/run.sh" "$@"
else
    echo "Error: ML Engine directory not found at $ENGINE_DIR"
    exit 1
fi
