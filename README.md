<div align="center">

# 🎙️ VoxFlow

**Voice-First Clarity for Everyday Work — 100% On-Device Speech Recognition, Adaptive Memory & Zero-Hallucination Refinement**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Whisper ASR](https://img.shields.io/badge/ASR-faster--whisper%20INT8-blueviolet)](https://github.com/SYSTRAN/faster-whisper)
[![Silero VAD](https://img.shields.io/badge/VAD-Silero%20ONNX-orange)](https://github.com/snakers4/silero-vad)
[![Tests Passing](https://img.shields.io/badge/Tests-102%20Passing-brightgreen?logo=pytest&logoColor=white)](voxflow/services/ml-engine/tests)
[![Type Checked](https://img.shields.io/badge/Type%20Check-Pyrefly%20Clean-success)](https://pyrefly.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[Quick Start](#-quick-start) • [Architecture](#-repository-architecture) • [Features](#-key-capabilities) • [Benchmarks](#-benchmarks--evaluation) • [Desktop App](#-native-desktop-overlay) • [API Guide](#-api--websocket-reference) • [Security](#-privacy--security)

</div>

---

## 📌 About VoxFlow

**VoxFlow** is an enterprise-grade, voice-first productivity platform that turns rambling, unstructured speech into crisp, structured notes and actionable summaries.

Unlike traditional dictation software that forces users to speak like a robot or cloud-dependent solutions that suffer from network latency and privacy leaks, VoxFlow runs **100% locally on your machine**:
- **Zero Cloud LLM Dependency**: No API keys required, zero per-query fees, and zero outbound network traffic during transcription.
- **Self-Learning Adaptive Memory**: Dynamically observes personal verbal filler habits (*"um"*, *"basically"*, *"like"*, *"yaar"*) and suppresses them automatically once confidence thresholds are reached.
- **Multilingual & Code-Switching**: Native support for English, Urdu script, and Roman Urdu code-switching without language suppression.
- **Smart Listener with Intent Guardrails**: Resolves mid-sentence self-corrections (*"Monday no wait Tuesday"* $\rightarrow$ *"Tuesday"*) and detects cancellation intents (*"cancel that just type hi"* $\rightarrow$ *"Hi"*).

---

## 🚀 Quick Start

### 1-Click Installation & Startup (Linux / macOS)

From the project root:

```bash
# 1. Run the 1-click installer (sets up Python virtualenv & dependencies)
./install.sh

# 2. Launch the ML Engine & Studio
./run.sh
```

### Windows (PowerShell / Command Prompt)

```cmd
:: 1-Click Install
install.bat

:: 1-Click Launch
run.bat
```

Once launched, access:
- **Interactive Web Studio**: [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Fluid Floating Pill Widget**: [http://127.0.0.1:8000/widget](http://127.0.0.1:8000/widget)
- **Interactive API Documentation**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Live Health Diagnostics**: [http://127.0.0.1:8000/health?details=true](http://127.0.0.1:8000/health?details=true)

---

## 🗂️ Repository Architecture

```
codep1/
├── install.sh / install.bat    # Root 1-click dependency installer
├── run.sh / run.bat            # Root 1-click launcher and test runner
├── eval_harness.py             # Accuracy, intent & code-switching evaluation harness
├── benchmark_latency.py        # Tier 1 (<0.2ms) & Tier 2 pipeline latency profiler
├── hygiene_audit.py            # Automated dead-code & credential hygiene auditor
├── pyrefly.toml                # Static type analyzer configuration
├── SECURITY.md                 # Security, injection defense & privacy specifications
├── CHANGELOG.md                # Detailed release notes across versions
└── voxflow/
    ├── apps/
    │   └── desktop/            # Cross-platform Electron desktop overlay (wake-word, auto-paste)
    ├── services/
    │   └── ml-engine/          # FastAPI ML Engine, VAD, ASR, Adaptive Memory & Web Studio
    ├── data/                   # Benchmark datasets, correction logs & custom vocabulary
    ├── models/                 # Model artifacts & weights
    └── docs/                   # Architectural notes & specs
```

---

## ✨ Key Capabilities

### 1. Local Speech Recognition & Silero VAD
- **`faster-whisper` (CTranslate2 INT8)** on CPU with multilingual code-switching support.
- **Silero Voice Activity Detection (ONNX)** with $<4\text{ms}$ p95 frame latency, acoustic fallback, and real-time silence duration estimation.
- Decoder prompt biasing (`initial_prompt`) fed automatically from your personal technical vocabulary.

### 2. Self-Learning Adaptive Memory (`adaptive_memory.py`)
- Tracks user-specific verbal filler frequencies over time.
- Suppresses habitual fillers automatically once crossed observation thresholds ($\ge 3$ hits).
- Mins domain-specific bigrams (*"staging database"*, *"pull request"*) to protect technical jargon.
- Atomic, crash-safe state updates using temporary-file swaps.

### 3. Context-Aware Local Cleanup (`local_cleanup_service.py`)
- **Self-Correction Pruning**: Detects conversational pivot markers (*"no wait"*, *"scratch that"*, *"actually"*) and retains only the finalized thought.
- **Comparison Protection**: Preserves meaningful usages of words like *"like"* (*"looks like an issue"* is kept; *"we should like deploy"* is cleaned).
- **Sequential Structuring**: Detects multi-step tasks and formats clean Markdown numbered action items.

### 4. Smart Listener & Intent Finalization (`intent_engine.py`)
- **Implicit Commits**: Evaluates silence duration ($\ge 1200\text{ms}$) after complete grammatical clauses.
- **Trailing Hold**: Extends pause tolerance ($2200\text{ms}$) when trailing conjunctions (*"and..."*, *"but..."*, *"lekin..."*) indicate unfinished thoughts.
- **Conversational Discard**: Recognizes cancellation phrases (*"cancel that"*, *"rehne do"*, *"chodo"*) and wipes exploratory speech without pasting.

---

## 📊 Benchmarks & Evaluation

Run the automated evaluation and latency benchmarks locally:

```bash
# Evaluate accuracy, intent preservation, and code-switching (14 test scenarios)
python eval_harness.py

# Benchmark pipeline latency across Tier 1 (interim) and Tier 2 (finalized)
python benchmark_latency.py

# Run comprehensive hygiene and dead-code audit
python hygiene_audit.py
```

### Verified Benchmark Results (Local CPU):

| Metric | Target | Measured Result | Status |
| :--- | :--- | :--- | :--- |
| **Eval Harness Pass Rate** | 100% | **14 / 14 (100.0%)** | ✅ PASS |
| **False-Cancel Rate** | 0.0% | **0.0%** | ✅ OPTIMAL |
| **False-Commit Rate** | 0.0% | **0.0%** | ✅ OPTIMAL |
| **Meaning Preservation** | $\ge 0.85$ | **0.928 / 1.000** | ✅ PASS |
| **Tier 1 Latency (p95)** | $< 2.0\text{ ms}$ | **0.089 ms** | ✅ PASS |
| **Tier 2 Latency (p95)** | Local On-Device | **0.096 ms** | ✅ PASS |
| **VAD Frame Latency (p95)** | $< 4.0\text{ ms}$ | **1.82 ms** | ✅ PASS |
| **Outbound Network Sockets** | 0 sockets | **0 sockets (100% Offline)** | ✅ AUDITED |

---

## 💻 Native Desktop Overlay (`voxflow/apps/desktop`)

VoxFlow includes a native desktop overlay with automatic application pasting:
- **Summon via Wake-Word or Hotkey**: Say *"VoxFlow"* or press `Ctrl+Shift+Space` (`Cmd+Shift+Space` on macOS).
- **Cylindrical Fluid Bar**: Glassmorphic floating bar with reactive frequency visualizer.
- **Intelligent Auto-Paste**: Automatically switches focus and pastes cleaned speech directly into your active window (IDE, Slack, browser, document editor).
- **Security Guardrail**: Protects sensitive password prompts (`sudo`, `1password`, `bitwarden`, `keepass`) from accidental injection.

### Starting the Desktop Shell:

```bash
cd voxflow/apps/desktop
npm install
npm start
```

For packaging desktop binaries (`.AppImage`, `.deb`, `.exe`, `.dmg`), refer to [apps/desktop/INSTALL.md](voxflow/apps/desktop/INSTALL.md).

---

## 📡 API & WebSocket Reference

### Real-Time WebSocket: `ws://127.0.0.1:8000/ws`

| Message Type | Direction | Payload Example | Description |
| :--- | :--- | :--- | :--- |
| `ping` | Client $\rightarrow$ Server | `{"type": "ping"}` | Keep-alive heartbeat (returns `pong`) |
| `transcript_interim` | Client $\rightarrow$ Server | `{"type": "transcript_interim", "text": "um we should"}` | Instant preview cleaning ($<0.2\text{ms}$) |
| `transcript` | Client $\rightarrow$ Server | `{"type": "transcript", "text": "...", "language": "auto"}` | Full multi-stage refinement |
| `job_submit` | Client $\rightarrow$ Server | `{"type": "job_submit", "text": "..."}` | Enqueue asynchronous background job |

### REST Endpoints

- `GET /health` / `GET /api/v1/health` — Service liveness and detailed component metrics.
- `POST /api/v1/refine/sync` — Synchronous transcript refinement with multi-language and formatting options.
- `POST /api/v1/jobs/refine` — Asynchronous background job submission (returns `202 Accepted` with tracking ID).
- `GET /api/v1/jobs/{job_id}` — Query status and result of background refinement jobs.
- `GET /api/v1/jobs/dead-letter` — Inspect dead-letter queue for exhausted retry attempts.
- `GET /memory` — Real-time inspect of adaptive memory, learned fillers, and domain bigrams.
- `GET /api/v1/personalization/metrics` — Adaptation trends and user edit rate history.
- `POST /api/v1/personalization/vocabulary` — Add custom technical terms or proper nouns to ASR bias list.

---

## 🧪 Testing & Code Quality

VoxFlow maintains 100% automated test suite pass rate:

```bash
# Run complete test suite (102 tests)
./run.sh pytest

# Run with verbose output
cd voxflow/services/ml-engine
pytest tests -v

# Run type checker
uvx pyrefly check
```

---

## 🔒 Privacy & Security

- **Air-Gapped Operation**: Dictation, intent processing, and punctuation run completely on-device without telemetry or cloud transmissions.
- **Local Personalization**: Correction logs (`corrections.jsonl`) and vocabulary (`personal_vocabulary.json`) remain strictly in local storage.
- **Injection Audit**: Detailed injection logging (`injection_audit.jsonl`) guarantees transparency of all clipboard and paste operations.

For full vulnerability disclosure and boundaries, see [SECURITY.md](SECURITY.md).

---

## 📄 License

Licensed under the [MIT License](LICENSE).
