import asyncio
import json
import logging
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Any

# ── Constants ─────────────────────────────────────────────────────────────────

log = logging.getLogger(__name__)

# A seed filler must be observed at least this many times before it is
# actively suppressed in future transcripts.
FILLER_THRESHOLD = 3

# Only words in this seed set can ever enter ``learned_fillers``.
# Content vocabulary is tracked separately and is *never* stripped.
SEED_FILLERS: frozenset[str] = frozenset({
    "um", "uh", "er", "ah", "like", "know", "okay", "ok",
    "alright", "right", "so", "well", "basically", "literally",
    "honestly", "actually", "obviously", "clearly", "anyway",
    "just", "really", "very", "quite", "rather",
})


# Informal-to-formal normalization pairs (pattern, replacement).
_NORM_PAIRS: list[tuple[str, str]] = [
    (r"\bgonna\b",   "going to"),
    (r"\bwanna\b",   "want to"),
    (r"\bgotta\b",   "got to"),
    (r"\bkinda\b",   "kind of"),
    (r"\bsorta\b",   "sort of"),
    (r"\bdunno\b",   "don't know"),
    (r"\bcoulda\b",  "could have"),
    (r"\bwoulda\b",  "would have"),
    (r"\bshoulda\b", "should have"),
    (r"\blemme\b",   "let me"),
    (r"\bgimme\b",   "give me"),
    (r"\byeah\b",    "yes"),
    (r"\bnope\b",    "no"),
]

# Pre-compile for speed.
_COMPILED_NORMS = [
    (re.compile(pat, re.IGNORECASE), rep) for pat, rep in _NORM_PAIRS
]


# ── Class ─────────────────────────────────────────────────────────────────────

class AdaptiveMemory:
    def __init__(self, storage_path: str | Path | None = None):
        self.storage_path = Path(
            storage_path or Path(__file__).with_name("adaptive_memory.json")
        )
        self._lock = threading.Lock()
        self.stats: dict[str, Any] = {
            "transcript_count": 0,
            "learned_fillers": {},   # seed-filler word → observation count
            "phrase_counts": {},     # bigram → observation count (persisted)
            "common_phrases": [],    # top-8 bigrams derived from phrase_counts
        }
        self.load()

    # ── I/O ───────────────────────────────────────────────────────────────────

    def load(self) -> None:
        if not self.storage_path.exists():
            return
        with self._lock:
            try:
                data = json.loads(self.storage_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Failed to load %s: %s", self.storage_path, exc)
                return
            self.stats.update(data)
            if not isinstance(self.stats.get("learned_fillers"), dict):
                self.stats["learned_fillers"] = {}
            if not isinstance(self.stats.get("phrase_counts"), dict):
                self.stats["phrase_counts"] = {}
            if not isinstance(self.stats.get("common_phrases"), list):
                self.stats["common_phrases"] = []

    def save(self) -> None:
        """Persist stats to disk using an atomic write (tmp + rename)."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = json.dumps(self.stats, indent=2)
        # Write to a temporary file in the same directory, then rename so a
        # crash mid-write never corrupts the JSON.
        fd, tmp_path = tempfile.mkstemp(
            dir=self.storage_path.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp_path, self.storage_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    async def async_save(self) -> None:
        """Asynchronously persist stats to disk in a separate thread."""
        await asyncio.to_thread(self.save)

    def reset(self) -> None:
        """Clear all learned state and remove the persisted file."""
        with self._lock:
            self.stats = {
                "transcript_count": 0,
                "learned_fillers": {},
                "phrase_counts": {},
                "common_phrases": [],
            }
        if self.storage_path.exists():
            try:
                self.storage_path.unlink()
            except OSError:
                pass
        log.info("Adaptive memory reset")

    # ── Learning ──────────────────────────────────────────────────────────────

    def observe(self, transcript: str) -> None:
        """Record a raw transcript for adaptive learning.

        Only seed filler words accumulate in ``learned_fillers``.
        Content vocabulary is tracked via bigrams in ``phrase_counts``.
        """
        if not transcript or not transcript.strip():
            return
        with self._lock:
            self.stats["transcript_count"] = int(self.stats.get("transcript_count", 0)) + 1
            words = re.findall(r"[a-zA-Z']+", transcript.lower())

            # Track only seed filler words — never pollute with content vocabulary.
            learned_fillers: dict[str, int] = self.stats.setdefault("learned_fillers", {})
            for word in words:
                if word in SEED_FILLERS:
                    learned_fillers[word] = learned_fillers.get(word, 0) + 1

            # Accumulate bigram phrase counts across all observations.
            phrase_counts: dict[str, int] = self.stats.setdefault("phrase_counts", {})
            content = [w for w in words if len(w) > 2 and w not in SEED_FILLERS]
            for i in range(len(content) - 1):
                bigram = f"{content[i]} {content[i + 1]}"
                phrase_counts[bigram] = phrase_counts.get(bigram, 0) + 1

            self.stats["common_phrases"] = self._top_phrases()

    # ── Processing ────────────────────────────────────────────────────────────

    @staticmethod
    def normalize(transcript: str) -> str:
        """Apply informal-to-formal normalization (gonna→going to, etc.)."""
        for pattern, replacement in _COMPILED_NORMS:
            transcript = pattern.sub(replacement, transcript)
        return transcript

    def adapt_transcript(self, transcript: str) -> str:
        """Normalize informal speech then strip high-frequency personal fillers."""
        # 1. Informal normalization
        cleaned = self.normalize(transcript.strip())
        cleaned = re.sub(r"\s+", " ", cleaned)

        # 2. Strip only seed fillers that have crossed the frequency threshold.
        learned: dict[str, int] = self.stats.get("learned_fillers", {})
        _EXCLUDED_PRONOUNS = {"you", "i", "we", "he", "she", "they", "me", "us", "him", "her", "them"}
        active_fillers = {
            word for word, count in learned.items()
            if count >= FILLER_THRESHOLD and word.lower() not in _EXCLUDED_PRONOUNS
        }

        for word in sorted(active_fillers, key=len, reverse=True):
            if word.lower() == "like":
                # Protect comparison 'like' (e.g. 'sounds like', 'like a Tesla')
                cleaned = re.sub(
                    r"\b(look|looks|sound|sounds|feel|feels|seem|seems|is|are|was|were|be|been|would|something)\s+like\b",
                    r"\1 __COMP_LIKE__",
                    cleaned,
                    flags=re.IGNORECASE,
                )
                cleaned = re.sub(
                    r"\blike\s+(a|an|the|this|that|these|those|how|what)\b",
                    r"__COMP_LIKE__ \1",
                    cleaned,
                    flags=re.IGNORECASE,
                )
                # Strip remaining hesitation 'like'
                cleaned = re.sub(r"\blike\b", "", cleaned, flags=re.IGNORECASE)
                cleaned = cleaned.replace("__COMP_LIKE__", "like")
            else:
                cleaned = re.sub(
                    rf"\b{re.escape(word)}\b", "", cleaned, flags=re.IGNORECASE
                )

        # 3. Tidy up whitespace and punctuation.
        cleaned = re.sub(r"\s+([,.;:?!])", r"\1", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _top_phrases(self) -> list[str]:
        """Return the top-8 most frequently observed content bigrams."""
        phrase_counts: dict[str, int] = self.stats.get("phrase_counts", {})
        if not phrase_counts:
            return []
        return [
            phrase
            for phrase, _ in sorted(
                phrase_counts.items(), key=lambda x: x[1], reverse=True
            )[:8]
        ]

    def summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "transcript_count": int(self.stats.get("transcript_count", 0)),
                "learned_fillers": dict(self.stats.get("learned_fillers", {})),
                "common_phrases": list(self.stats.get("common_phrases", [])),
            }
