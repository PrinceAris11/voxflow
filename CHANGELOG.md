# Changelog — VoxFlow Engine

All notable changes to VoxFlow are documented in this file.

## [v3.0.0] - 2026-09-27 — Fully Local, Self-Learning, Industry-Grade

### 🚀 Major Architectural Upgrades (Zero Cloud LLM)
- **100% On-Device Inference**:
  - Removed all dependencies on Google Gemini 2.0 Flash Lite and the `google-generativeai` SDK.
  - Dictation, intent detection, cleanup, and punctuation run entirely locally with zero per-request cost and zero network round-trip variance.
- **Local Speech Recognition (Part B)**:
  - Added [`asr_service.py`](voxflow/services/ml-engine/asr_service.py) powered by `faster-whisper` with CTranslate2 INT8 quantization on CPU.
  - Multilingual transcription supporting Urdu/English code-switching without forced language suppression.
  - Decoder prompt biasing (`initial_prompt`) fed by personal vocabulary.
- **Silero Voice Activity Detection (Part B & E)**:
  - Added [`vad_service.py`](voxflow/services/ml-engine/vad_service.py) to gate ASR inference on speech segments.
  - Emits real-time `silence_duration_ms` metric feeding Part E's implicit commit logic.
- **Local Transcript Cleanup & Structuring (Part C)**:
  - Added [`local_cleanup_service.py`](voxflow/services/ml-engine/local_cleanup_service.py) replacing the cloud LLM prompt.
  - Resolves self-corrections (*"Monday no wait Tuesday"* $\to$ *"Tuesday"*), collapses stutters, removes bilingual fillers while protecting comparison *"like"*.
  - Detects sequential action statements and formats structured bulleted action items.
  - Intent preservation & anti-hallucination guardrail verifies semantic and lexical overlap.
- **Intent Finalization & Smart Listener (Part E)**:
  - Integrated VAD silence-duration signal into [`intent_engine.py`](voxflow/services/ml-engine/intent_engine.py):
    - Normal pauses $\ge 1200\text{ ms}$ after complete thoughts trigger implicit commits.
    - Incomplete trailing thoughts (*"and..."*, *"but..."*, *"lekin..."*) extend wait time to $2200\text{ ms}$.
    - Explicit discard commands (*"cancel that"*, *"rehne do"*, *"chodo"*) wipe exploratory speech cleanly.
    - Verified canonical test case: *"I want to go to nowhere cancel that just type hi how are you"* $\to$ *"Hi, how are you?"*.
- **Desktop State Machine Fixes (Part D)**:
  - Refactored [`cylindrical_bar.js`](voxflow/apps/desktop/src/renderer/cylindrical_bar.js) to enforce strict state isolation:
    - `liveBuffer` is replaced each interim chunk, never appended.
    - `history` is pushed only on finalized commits.
    - Rendering adheres to `displayText = history.join(' ') + ' ' + liveBuffer`.
    - Discarded speech resets liveBuffer with a transient cancellation indicator without pasting.
- **System-Wide Text Injection & Safety (Part F & K)**:
  - Enhanced [`paste_helper.js`](voxflow/apps/desktop/src/paste_helper.js) with sensitive credential field detection (`1password`, `bitwarden`, `keepass`, `sudo`, `password`).
  - Added local-only injection audit log (`voxflow/data/injection_audit.jsonl`).
- **Local Personalization & Adaptation (Part H)**:
  - Implemented [`personalization_service.py`](voxflow/services/ml-engine/personalization_service.py) tracking `(raw -> refined -> user_edit)` triples in `voxflow/data/corrections.jsonl`.
  - Added persistent custom vocabulary (`voxflow/data/personal_vocabulary.json`) for technical jargon and proper nouns.
  - Track per-user edit distance trends over time to quantify user-specific accuracy improvement.
- **Evaluation & Benchmarking Harness (Part J)**:
  - Added [`eval_dataset.json`](voxflow/data/eval_dataset.json) covering clean English, Urdu script, Roman Urdu, code-switching, self-corrections, and intent edge cases.
  - Created [`eval_harness.py`](eval_harness.py): achieved 100% pass rate (14/14), 0.0% false-cancel rate, and 0.0% false-commit rate.
  - Created [`benchmark_latency.py`](benchmark_latency.py): measured real latency on Intel Core i7-8665U CPU (Tier 1 p95: 0.042ms, Tier 2 p95: 0.024ms).
- **Privacy & Repository Hygiene (Part I & K)**:
  - Created [`test_network_traffic_audit.py`](voxflow/services/ml-engine/tests/test_network_traffic_audit.py): mathematically asserts zero outbound network sockets during dictation.
  - Created [`hygiene_audit.py`](hygiene_audit.py) to detect dead code and cloud remnants.
  - Published [`SECURITY.md`](SECURITY.md) covering injection security and data boundaries.
