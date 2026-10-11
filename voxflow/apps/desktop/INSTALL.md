# 🎙️ VoxFlow Desktop Installation & Build Guide

VoxFlow Desktop is a native cross-platform assistant for **Windows, macOS, and Linux**. It features a floating, transparent cylindrical bar that listens when summoned with the wake-word **"VoxFlow"** (or global hotkey `Ctrl+Shift+Space` / `Cmd+Shift+Space`), refines speech through VoxFlow's self-learning **Adaptive Memory** engine, and automatically pastes the result directly into your active application.

---

## 📋 Prerequisites

1. **Node.js**: Version 18.x, 20.x, or later ([Download Node.js](https://nodejs.org/))
2. **Python**: Version 3.10+ (for the VoxFlow ML Engine backend)

---

## ⚡ Quick Start (Development Mode)

### Step 1: Start the ML Engine Backend
In a terminal, start the VoxFlow backend service:
```bash
cd voxflow/services/ml-engine
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
python main.py
```
The ML Engine starts at `http://127.0.0.1:8000` with the WebSocket endpoint at `ws://127.0.0.1:8000/ws`.

### Step 2: Launch the Desktop App
In a second terminal:
```bash
cd voxflow/apps/desktop
npm install
npm start
```
The floating cylindrical bar will appear on your desktop.

---

## 🛠️ Building Native Installers

VoxFlow uses `electron-builder` to package native installers for all major operating systems.

### 🪟 Windows (`.exe` Installer)
To build the standalone Windows installer (NSIS) and portable executable:
```bash
npm run dist:win
```
- **Output**: `apps/desktop/dist/VoxFlow Setup 1.0.0.exe` and `apps/desktop/dist/VoxFlow 1.0.0.exe`
- **Auto-Paste**: Works out of the box using Windows PowerShell / SendKeys automation.

### 🍎 macOS (`.dmg` Installer)
To build the macOS `.dmg` package:
```bash
npm run dist:mac
```
- **Output**: `apps/desktop/dist/VoxFlow-1.0.0.dmg`
- **macOS Permissions**: 
  - To enable auto-pasting into other applications, grant **Accessibility** permissions:
    - Open **System Settings** > **Privacy & Security** > **Accessibility**.
    - Add or toggle **VoxFlow** to allowed.

### 🐧 Linux (`.AppImage` & `.deb`)
To build the Linux packages:
```bash
npm run dist:linux
```
- **Output**: 
  - `apps/desktop/dist/VoxFlow-1.0.0.AppImage` (Universal Linux package)
  - `apps/desktop/dist/voxflow-desktop_1.0.0_amd64.deb` (Debian/Ubuntu/Mint)
- **Wayland / X11 Keystroke Automation**:
  - On **X11**: `xdotool` is supported for automatic `Ctrl+V` keystroke injection (`sudo apt install xdotool`).
  - On **Wayland**: `wtype` or `ydotool` can be used (`sudo apt install wtype`).
  - If external injectors are absent, VoxFlow safely places the refined text onto the clipboard ready for immediate manual `Ctrl+V`.

---

## 💡 How to Use

1. **Summoning VoxFlow**:
   - **Voice Wake-Word**: Speak *"VoxFlow"* or *"Hey VoxFlow"*. The cylindrical bar activates and expands.
   - **Global Shortcut**: Press `Ctrl+Shift+Space` (or `Cmd+Shift+Space` on Mac) anywhere on your computer.
   - **Manual Click**: Click anywhere on the cylindrical bar.
2. **Speak Naturally**:
   - Speak your thoughts freely with pauses or filler words (*"Um like let's push the release to Monday"*).
3. **Automatic Refinement & Paste**:
   - When you pause for ~1.8 seconds, VoxFlow sends your speech to the Adaptive Memory engine.
   - The engine strips filler words, corrects contractions, refines grammar, and learns your speech patterns.
   - The floating bar hides, restores focus to your active text box, and automatically pastes the cleaned text!
