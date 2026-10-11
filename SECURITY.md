# Security & Privacy Architecture — VoxFlow v3

## 1. Zero Cloud Dependency & Offline Guarantee
VoxFlow v3 is engineered as a **100% on-device, zero-cloud** voice dictation engine.
- **Zero Third-Party Telemetry**: VoxFlow ships without any remote analytics, tracking pixels, or cloud reporting SDKs.
- **Zero Remote Inference**: Audio capture, Silero Voice Activity Detection (VAD), Speech-to-Text (ASR), Intent Finalization, and Language Identification execute exclusively within the local host process.
- **Network Traffic Audit Verification**: The test suite includes [`test_network_traffic_audit.py`](voxflow/services/ml-engine/tests/test_network_traffic_audit.py), which intercepts all outbound socket connection attempts during full dictation sessions and programmatically verifies that **zero outbound network calls** occur.

---

## 2. OS-Level Injection Safety & Permissions

### Threat Model: System-Wide Text Injection
Because VoxFlow possesses OS-level text injection capability (injecting finalized text into whichever application currently has user focus), it treats synthetic keystroke automation with strict security discipline:
1. **Sensitive Target Detection**:
   - `PasteHelper.isSensitiveTarget(targetTitle)` inspects window and app titles against known credential and password manager signatures (`1password`, `bitwarden`, `keepass`, `lastpass`, `dashlane`, `sudo`, `terminal - su`, `credential`).
   - When a sensitive window is detected, synthetic keystroke injection is **aborted**, and the text remains exclusively in the user's private clipboard with a notification.
2. **Focus-Change Abort**:
   - If user focus changes mid-utterance, text injection aborts to avoid spraying text into an unintended application window.
3. **Local Injection Audit Log**:
   - Injection actions are logged locally in `voxflow/data/injection_audit.jsonl` containing only timestamp, character count, method, and safety status.
   - Raw sensitive text is not logged in plaintext.
   - Users can purge this audit log anytime.

---

## 3. Local Personalization & Adaptive Data Handling

- **Data at Rest**:
  - Personal vocabulary (`voxflow/data/personal_vocabulary.json`) and correction triples (`voxflow/data/corrections.jsonl`) reside strictly on the local filesystem.
  - No user corrections, vocabulary terms, or audio recordings are ever transmitted off the user's device.
- **Safe Adapter Training**:
  - Personalization uses Low-Rank Adaptation (LoRA) adapters with versioning and rollback.
  - Adaptation processes run only during designated idle periods to ensure zero interference with foreground compute.

---

## 4. Credential & Key Hygiene
- The legacy cloud Google Gemini integration has been fully removed.
- VoxFlow v3 requires **no API keys** and no cloud credentials.
- Users upgrading from v1/v2 are advised to revoke any old `GEMINI_API_KEY` created in Google AI Studio and delete it from `.env`.
