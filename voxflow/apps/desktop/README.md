# 🎙️ VoxFlow Desktop

Cross-platform desktop companion featuring a floating cylindrical overlay, voice wake-word activation, real-time Adaptive Memory refinement, and automatic clipboard copy and paste into active applications.

---

## ✨ Features

- **🛸 Floating Cylindrical Bar**: Frameless, transparent, always-on-top capsule with glowing reactive audio visualizer.
- **🗣️ Wake-Word Summoning**: Activated by saying *"VoxFlow"* or *"Hey VoxFlow"*, or using the global shortcut `Ctrl+Shift+Space` (`Cmd+Shift+Space` on macOS).
- **📋 Automatic Copy & Paste**: Automatically copies the refined text and simulates the native paste shortcut (`Ctrl+V` / `Cmd+V`) directly into whichever application or text field you were using.
- **🧠 Continuous Self-Learning**: Seamlessly streams to the VoxFlow ML Engine over WebSockets (`ws://127.0.0.1:8000/ws`). Automatically learns and prunes your personal filler words over time.
- **💻 Cross-Platform Installers**: Configured with `electron-builder` for **Windows** (`.exe`), **macOS** (`.dmg`), and **Linux** (`.AppImage`, `.deb`).

---

## 🚀 Quick Start

### 1. Start the ML Engine Backend
```bash
cd ../../services/ml-engine
source .venv/bin/activate
python main.py
```

### 2. Run the Desktop App
```bash
cd ../../apps/desktop
npm install
npm start
```

For complete packaging and build commands, see [INSTALL.md](INSTALL.md).
