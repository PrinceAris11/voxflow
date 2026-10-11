"""intent_engine.py — VoxFlow Intent Finalization & "Smart Listener" Engine.

Differentiates between exploratory "thinking out loud" speech vs. the actual
intended message the user wants output/typed.

Unlike Part C (which cleans grammar/fillers inside a single intended thought),
Part G decides WHICH portion of a longer, meandering utterance is the genuine
intent, discarding exploratory rambling entirely.

Key Capabilities:
1. EXPLICIT DISCARD SIGNALS:
   - English: "cancel that", "nevermind", "scratch this", "forget it", "delete that", "ignore that"
   - Roman Urdu: "chodo", "chhoro", "rehne do", "hatao isko", "nahi rehne do"
   - Urdu Script: "چھوڑو", "رہنے دو", "منسوخ"
   -> Action: Wipes everything spoken before the discard signal.
2. EXPLICIT COMMIT SIGNALS:
   - English: "just type", "type this", "write down", "just write", "send this", "type:"
   - Roman Urdu: "sirf ye likho", "bas ye likho", "likho ke", "ye type karo"
   - Urdu Script: "صرف یہ لکھو", "یہ لکھو", "ٹائپ کرو"
   -> Action: Isolates and extracts only the payload following the commit phrase.
3. INCOMPLETE THOUGHT & CONFIDENCE-GATING:
   - Detects trailing conjunctions ("and...", "because...", "lekin...", "aur...")
   - Detects dangling prepositions ("to...", "about...", "mein...")
   - Distinguishes complete statements from tentative trails.
4. INTENT BUFFER ARCHITECTURE:
   - Maintains an `intentBuffer` across continuous fragments.
   - On DISCARD: `intentBuffer = ""` (exploratory rambling thrown away).
   - On COMMIT: passes only the confirmed intent to downstream pipeline (C -> D -> E).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class IntentAction(str, Enum):
    COMMIT = "commit"        # Confirmed intent ready for full pipeline & injection
    DISCARD = "discard"      # Cancelled / exploratory rambling wiped out
    PENDING = "pending"      # Incomplete or trailing thought; keep listening
    PASS_THROUGH = "pass"    # Standard speech without explicit commands


# ─── Explicit Discard & Cancel Signals (Multi-Lingual) ─────────────────────────

_DISCARD_PATTERNS = [
    # English
    re.compile(r"\b(?:cancel\s+(?:that|this|it)|never\s*mind|scratch\s+(?:that|this)|forget\s+(?:it|that|this)|ignore\s+(?:that|this)|delete\s+(?:that|this)|throw\s+(?:it|that)\s+away)\b", re.IGNORECASE),
    # Roman Urdu
    re.compile(r"\b(?:chhoro\s*(?:isko|ye)?|chodo\s*(?:isko|ye)?|choro\s*(?:isko|ye)?|rehne\s+do|hatao\s+isko|nahi\s+rehne\s+do|isko\s+khatam\s+karo)\b", re.IGNORECASE),
    # Urdu Script
    re.compile(r"[\u0600-\u06FF]*(?:چھوڑو|رہنے\s+دو|منسوخ|ہٹاؤ)[\u0600-\u06FF]*"),
]


# ─── Explicit Commit / Focus Signals (Multi-Lingual) ──────────────────────────

_COMMIT_PATTERNS = [
    # English
    re.compile(r"\b(?:just\s+type\s*:?|type\s+this\s*:?|just\s+write\s*:?|write\s+down\s*:?|send\s+this\s*:?|put\s+this\s+down\s*:?|type\s*:)\s*", re.IGNORECASE),
    # Roman Urdu
    re.compile(r"\b(?:sirf\s+ye\s+likho\s*:?|bas\s+ye\s+likho\s*:?|likho\s+ke\s*:?|ye\s+type\s+karo\s*:?|sirf\s+likho\s*:?)\s*", re.IGNORECASE),
    # Urdu Script
    re.compile(r"[\u0600-\u06FF]*(?:صرف\s+یہ\s+لکھو|یہ\s+لکھو|ٹائپ\s+کرو)[\u0600-\u06FF]*\s*"),
]


# ─── Dangling / Incomplete Trailing Markers ─────────────────────────────────────

_TRAILING_INCOMPLETE = re.compile(
    r"\b(?:and|but|or|so|because|although|though|if|while|since|that|which|to|with|for|about|into|onto|aur|lekin|magar|kyun\s*ke|agar|par|se|mein|اور|لیکن|اگر|کیونکہ)\s*$",
    re.IGNORECASE,
)


@dataclass
class IntentParseResult:
    action: IntentAction
    intended_text: str
    discarded_text: str | None = None
    confidence: float = 1.0
    command_detected: str | None = None
    is_complete: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "intended_text": self.intended_text,
            "discarded_text": self.discarded_text,
            "confidence": round(self.confidence, 3),
            "command_detected": self.command_detected,
            "is_complete": self.is_complete,
        }


class IntentEngine:
    """Smart Listener intent finalization and exploratory speech pruning engine."""

    def __init__(self, silence_commit_threshold: float = 1.2):
        self.silence_commit_threshold = silence_commit_threshold

    def parse_utterance(
        self,
        raw_text: str,
        language: str = "en",
        silence_duration_ms: float = 0.0,
    ) -> IntentParseResult:
        """Parse an utterance for macro discard signals, explicit commit triggers,
        incomplete thought markers, and VAD silence duration implicit commit.
        """
        text = raw_text.strip()
        if not text:
            return IntentParseResult(
                action=IntentAction.PASS_THROUGH,
                intended_text="",
                discarded_text=None,
                confidence=1.0,
                is_complete=True,
            )

        # 1. Check for explicit Commit markers ("just type ...", "bas ye likho ...")
        for pat in _COMMIT_PATTERNS:
            match = pat.search(text)
            if match:
                prefix = text[:match.start()].strip()
                payload = text[match.end():].strip()
                command = match.group(0).strip(" :")

                is_complete = not bool(_TRAILING_INCOMPLETE.search(payload))
                return IntentParseResult(
                    action=IntentAction.COMMIT,
                    intended_text=payload,
                    discarded_text=prefix if prefix else None,
                    confidence=0.95,
                    command_detected=command,
                    is_complete=is_complete,
                )

        # 2. Check for explicit Discard / Cancel markers ("cancel that", "rehne do")
        for pat in _DISCARD_PATTERNS:
            match = pat.search(text)
            if match:
                discard_point = match.end()
                discarded_chunk = text[:discard_point].strip()
                remaining = text[discard_point:].strip()
                command = match.group(0).strip()

                # Consume consecutive discard signals (e.g. "cancel that, nevermind", "choro isko rehne do")
                while remaining:
                    stripped = remaining.lstrip(" ,.-:;—")
                    matched_consecutive = False
                    for p2 in _DISCARD_PATTERNS:
                        m2 = p2.match(stripped)
                        if m2:
                            remaining = stripped[m2.end():].strip()
                            matched_consecutive = True
                            break
                    if not matched_consecutive:
                        break

                # Check if there is an explicit commit inside the remaining text
                for cp in _COMMIT_PATTERNS:
                    cm = cp.search(remaining)
                    if cm:
                        payload = remaining[cm.end():].strip()
                        return IntentParseResult(
                            action=IntentAction.COMMIT,
                            intended_text=payload,
                            discarded_text=text[:text.find(payload)].strip(),
                            confidence=0.95,
                            command_detected=cm.group(0).strip(" :"),
                            is_complete=not bool(_TRAILING_INCOMPLETE.search(payload)),
                        )

                # Check if substantive speech follows the cancel
                substantive = re.sub(r"^[,\s.:;—]+", "", remaining)
                if substantive and len(substantive.split()) >= 2:
                    substantive_clean = re.sub(r"^(?:now|acha|to|so|okay)\s*[,:]?\s*", "", substantive, flags=re.IGNORECASE)
                    is_complete = not bool(_TRAILING_INCOMPLETE.search(substantive_clean))
                    return IntentParseResult(
                        action=IntentAction.COMMIT,
                        intended_text=substantive_clean,
                        discarded_text=discarded_chunk,
                        confidence=0.90,
                        command_detected=command,
                        is_complete=is_complete,
                    )

                # Pure discard signal with nothing following
                return IntentParseResult(
                    action=IntentAction.DISCARD,
                    intended_text="",
                    discarded_text=text,
                    confidence=0.98,
                    command_detected=command,
                    is_complete=True,
                )


        # 3. Check for incomplete trailing thoughts vs VAD silence-duration signal
        has_trailing_incomplete = bool(_TRAILING_INCOMPLETE.search(text))
        is_complete = not has_trailing_incomplete

        # Tunable thresholds
        standard_silence_threshold = self.silence_commit_threshold * 1000.0  # e.g. 1200ms
        extended_silence_threshold = standard_silence_threshold + 1000.0     # e.g. 2200ms

        if has_trailing_incomplete:
            if silence_duration_ms >= extended_silence_threshold:
                # User paused long enough that they finished speaking despite trailing conjunction
                trimmed_text = _TRAILING_INCOMPLETE.sub("", text).strip()
                action = IntentAction.COMMIT
                return IntentParseResult(
                    action=action,
                    intended_text=trimmed_text if trimmed_text else text,
                    discarded_text=None,
                    confidence=0.88,
                    command_detected="vad_extended_pause_commit",
                    is_complete=True,
                )
            else:
                # Trailing conjunction with brief pause: hold in PENDING
                return IntentParseResult(
                    action=IntentAction.PENDING,
                    intended_text=text,
                    discarded_text=None,
                    confidence=0.60,
                    command_detected="trailing_incomplete_wait",
                    is_complete=False,
                )

        # Grammatically complete thought: check if VAD silence confirms implicit commit
        if silence_duration_ms >= standard_silence_threshold:
            action = IntentAction.COMMIT
            confidence = 0.95
            command = "vad_silence_implicit_commit"
        else:
            action = IntentAction.PASS_THROUGH
            confidence = 0.85
            command = None

        return IntentParseResult(
            action=action,
            intended_text=text,
            discarded_text=None,
            confidence=confidence,
            command_detected=command,
            is_complete=is_complete,
        )


class IntentSessionTracker:
    """Stateful intent buffer manager across streaming audio segments.

    Maintains `intentBuffer` across continuous fragments, wiping on cancel
    and committing only confirmed, finalized thoughts.
    """

    def __init__(self, engine: IntentEngine | None = None):
        self.engine = engine or IntentEngine()
        self.intent_buffer: list[str] = []
        self.discarded_history: list[str] = []
        self.last_action: IntentAction = IntentAction.PASS_THROUGH

    def update_interim(self, text: str) -> str:
        """Lightweight pass on live interim speech."""
        parse = self.engine.parse_utterance(text)
        if parse.action == IntentAction.DISCARD:
            return "[Discarded]"
        elif parse.action == IntentAction.COMMIT:
            return parse.intended_text
        return parse.intended_text

    def process_final(self, text: str, silence_duration_ms: float = 0.0) -> IntentParseResult:
        """Process finalized segment with VAD silence metric, updating intent buffer state."""
        parse = self.engine.parse_utterance(text, silence_duration_ms=silence_duration_ms)
        self.last_action = parse.action

        if parse.action == IntentAction.DISCARD:
            # Wipe buffer
            if self.intent_buffer:
                self.discarded_history.extend(self.intent_buffer)
                self.intent_buffer.clear()
            if parse.discarded_text:
                self.discarded_history.append(parse.discarded_text)
            return parse

        elif parse.action == IntentAction.COMMIT:
            # If explicit commit extracted payload, replace/commit buffer
            if parse.discarded_text:
                self.discarded_history.append(parse.discarded_text)
            self.intent_buffer.clear()
            if parse.intended_text:
                self.intent_buffer.append(parse.intended_text)
            return parse

        elif parse.action == IntentAction.PENDING:
            self.intent_buffer.append(parse.intended_text)
            return parse

        else:  # PASS_THROUGH
            self.intent_buffer.append(parse.intended_text)
            return parse

    def get_commit_payload(self) -> str:
        """Retrieve full committed text and clear buffer."""
        payload = " ".join(self.intent_buffer).strip()
        self.intent_buffer.clear()
        return payload

    def reset(self):
        """Reset session."""
        self.intent_buffer.clear()
        self.discarded_history.clear()
        self.last_action = IntentAction.PASS_THROUGH


# Global singleton instance
intent_engine = IntentEngine()
