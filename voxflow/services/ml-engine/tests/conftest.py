import sys
from pathlib import Path

# Automatically ensure the ml-engine directory is on sys.path during test runs
SERVICE_DIR = str(Path(__file__).resolve().parents[1])
if SERVICE_DIR not in sys.path:
    sys.path.insert(0, SERVICE_DIR)
