<div align="center">

# 🎙️ VoxFlow

**Voice-first clarity for everyday work — 100% on-device speech refinement and adaptive memory.**

[![FastAPI](https://img.shields.io/badge/FastAPI-ML%20Engine-009688?logo=fastapi&logoColor=white)](services/ml-engine)
[![Whisper ASR](https://img.shields.io/badge/ASR-faster--whisper%20INT8-blueviolet)](https://github.com/SYSTRAN/faster-whisper)
[![Silero VAD](https://img.shields.io/badge/VAD-Silero%20ONNX-orange)](https://github.com/snakers4/silero-vad)
[![Tests Passing](https://img.shields.io/badge/Tests-102%20Passing-brightgreen?logo=pytest&logoColor=white)](services/ml-engine/tests)
[![Type Checked](https://img.shields.io/badge/Type%20Check-Pyrefly%20Clean-success)](https://pyrefly.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](../LICENSE)

</div>

---

## 📌 About VoxFlow

VoxFlow is a voice-first productivity platform that turns rambling, unstructured speech into clear, actionable notes and structured summaries. It couples a low-latency desktop interface with an intelligent machine learning engine powered by self-learning adaptive memory, Silero VAD, and local Whisper ASR running 100% on-device with zero cloud dependencies.

---

## 🗂️ Repository Architecture

```
voxflow/
├── apps/
│   └── desktop/           # Electron desktop shell & hotkey audio flow
├── services/
│   └── ml-engine/         # FastAPI WebSocket engine, Adaptive Memory & Local ML Refiner
├── data/                  # Benchmark datasets, correction logs & custom vocabulary
├── models/                # Exported model weights & acoustic artifacts
├── training/              # Training notebooks & fine-tuning pipelines
└── docs/                  # Architectural specs & design notes
```

### Key Services

- **[ML Engine Service (`services/ml-engine`)](services/ml-engine/README.md)**:
  - **FastAPI WebSocket Service** for real-time audio and transcript streaming.
  - **Adaptive Memory Engine** that learns and prunes personal verbal fillers dynamically.
  - **Zero-Hallucination On-Device Refiner** with self-correction pruning and markdown action items.
  - **Interactive Web Studio UI** and standalone floating pill widget.
  - Comprehensive unit test suite (`pytest`, 102 tests passing) and type verification (`pyrefly`).

---

## 🚀 Quick Start (ML Engine & Studio)

```bash
# 1. Navigate to the ML Engine service
cd services/ml-engine

# 2. Set up virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Start the server (runs 100% locally with zero cloud API keys needed)
python main.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser to access the **VoxFlow Studio**, or [http://localhost:8000/widget](http://localhost:8000/widget) for the floating pill widget.

---

## 💻 Native Desktop Application (`apps/desktop`)

VoxFlow includes a native cross-platform desktop overlay for **Windows, macOS, and Linux**:
- **Summon via Wake-Word**: Say *"VoxFlow"* or press `Ctrl+Shift+Space` (`Cmd+Shift+Space` on macOS).
- **Floating Cylindrical Bar**: Sleek floating glassmorphic pill with organic reactive audio visualizer.
- **Auto-Paste**: Automatically copies refined transcripts and pastes them directly into your active application.
- **Continuous Learning**: Adapts to your voice and filters your personal filler words over time.

### Quick Start Desktop:
```bash
cd apps/desktop
npm install
npm start
```

For complete packaging and installer build instructions (Windows `.exe`, macOS `.dmg`, Linux `.AppImage`/`.deb`), see **[apps/desktop/INSTALL.md](apps/desktop/INSTALL.md)**.
