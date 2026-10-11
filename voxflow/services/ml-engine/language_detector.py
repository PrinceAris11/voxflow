"""language_detector.py — Language Identification (LID) & Code-Switching Engine.

Detects languages in spoken transcripts:
- English ('en')
- Urdu script ('ur', Arabic unicode range)
- Roman Urdu ('ur_roman', Pakistani conversational transliteration)
- Code-switched English & Urdu / Roman Urdu ('en_ur_mixed')

Provides:
- Segment-level boundary tagging for mid-utterance code-switching
- Confidence scoring with stateful fallback to most recent confirmed language
- Output script preference handling (e.g. Roman Urdu vs Urdu script)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# ─── Lexicons & Patterns ───────────────────────────────────────────────────────

# Urdu Unicode range (Arabic / Urdu block: U+0600–U+06FF)
_URDU_SCRIPT_PATTERN = re.compile(r"[\u0600-\u06FF]")

# Distinctive Roman Urdu lexical markers common in Pakistani speech (excluding English homographs like 'the', 'to', 'me')
_ROMAN_URDU_LEXICON = frozenset({
    "yaar", "yar", "mujhe", "mujhey", "tumhe", "tumhein", "humein", "humain",
    "karni", "karna", "karein", "karo", "karta", "karti", "hai", "hain", "tha",
    "thi", "acha", "achha", "theek", "thik", "matlab", "matlb", "yani",
    "yaani", "nahi", "nahin", "nahe", "ruko", "rok", "chhoro", "chorro", "choro",
    "bhai", "lekin", "shukriya", "hogaya", "hogaye", "gaya", "gaye", "gayi",
    "bohot", "boht", "kya", "kyun", "kyu", "kab", "kahan", "kaise", "apna",
    "apni", "apne", "wala", "wali", "wale", "batayein", "batao", "dekh", "dekho",
    "suno", "zarurat", "masla", "shayed", "agar", "warna", "phir", "pehle",
    "baad", "abhi", "koshish", "samajh", "aaj", "kal", "parson", "subah", "shaam",
    "aao", "jao", "bol", "bolo", "paas", "door", "saath", "dost", "kaam", "zaroor"
})

# Common English stopwords & structural markers
_ENGLISH_LEXICON = frozenset({
    "the", "and", "that", "this", "with", "have", "from", "into", "your",
    "about", "for", "will", "would", "should", "could", "can", "then", "there",
    "they", "their", "more", "was", "are", "but", "not", "its", "has", "had",
    "were", "what", "when", "who", "how", "why", "where", "which", "today",
    "tomorrow", "yesterday", "meeting", "submit", "file", "project", "code",
    "build", "test", "deploy", "server", "client", "office", "morning", "please",
    "thanks", "update", "task", "review", "email", "report", "call", "schedule"
})


@dataclass
class SegmentLanguage:
    text: str
    language: str
    confidence: float


@dataclass
class LanguageDetectionResult:
    language: str                  # 'en', 'ur', 'ur_roman', 'en_ur_mixed'
    confidence: float              # 0.0 to 1.0
    is_code_switched: bool         # True if utterance mixes English and Urdu
    output_script: str             # 'latin' (English / Roman Urdu) or 'arabic' (Urdu script)
    segments: list[SegmentLanguage] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "confidence": round(self.confidence, 3),
            "is_code_switched": self.is_code_switched,
            "output_script": self.output_script,
            "segments": [
                {
                    "text": s.text,
                    "language": s.language,
                    "confidence": round(s.confidence, 3),
                }
                for s in self.segments
            ],
        }


class LanguageDetector:
    """Stateful language detector with history fallback for smooth streaming."""

    def __init__(self, default_language: str = "en", output_script_pref: str = "roman_urdu"):
        self.default_language = default_language
        self.output_script_pref = output_script_pref
        self.last_confirmed_language = default_language

    def detect(
        self,
        text: str,
        output_script_pref: str | None = None,
    ) -> LanguageDetectionResult:
        """Analyze text and detect language, script, code-switching, and confidence."""
        clean_text = text.strip()
        pref = output_script_pref or self.output_script_pref

        if not clean_text:
            return LanguageDetectionResult(
                language=self.last_confirmed_language,
                confidence=1.0,
                is_code_switched=False,
                output_script="latin" if self.last_confirmed_language != "ur" else "arabic",
                segments=[],
            )

        # 1. Check for native Urdu script (Arabic unicode characters)
        urdu_char_count = len(_URDU_SCRIPT_PATTERN.findall(clean_text))
        total_chars = len(clean_text.replace(" ", ""))

        if total_chars > 0 and (urdu_char_count / total_chars) > 0.35:
            # Over 35% Urdu characters -> Urdu Script
            conf = min(1.0, (urdu_char_count / total_chars) + 0.2)
            self.last_confirmed_language = "ur"
            return LanguageDetectionResult(
                language="ur",
                confidence=conf,
                is_code_switched=False,
                output_script="arabic",
                segments=[SegmentLanguage(text=clean_text, language="ur", confidence=conf)],
            )

        # 2. Token-level analysis for Latin text (English vs Roman Urdu vs Mixed)
        words = re.findall(r"[a-zA-Z']+", clean_text.lower())
        if not words:
            # Punctuation/numbers only -> keep previous language
            return LanguageDetectionResult(
                language=self.last_confirmed_language,
                confidence=0.5,
                is_code_switched=False,
                output_script="latin",
                segments=[],
            )

        roman_urdu_matches = 0
        english_matches = 0
        total_recognized = 0

        # Segment tagging: split by natural clauses or word chunks
        segments: list[SegmentLanguage] = []
        clause_splits = re.split(r"([,.;:?!]|\b(?:and|lekin|or|ya|phir)\b)", clean_text)
        current_chunk = ""

        for part in clause_splits:
            part_str = part.strip()
            if not part_str:
                continue
            part_words = re.findall(r"[a-zA-Z']+", part_str.lower())
            if not part_words:
                current_chunk += (" " + part_str) if current_chunk else part_str
                continue

            chunk_ur = sum(1 for w in part_words if w in _ROMAN_URDU_LEXICON)
            chunk_en = sum(1 for w in part_words if w in _ENGLISH_LEXICON)

            if chunk_ur > chunk_en:
                chunk_lang = "ur_roman"
                chunk_conf = min(0.95, 0.5 + (chunk_ur / max(1, len(part_words))) * 0.45)
            elif chunk_en > chunk_ur:
                chunk_lang = "en"
                chunk_conf = min(0.95, 0.5 + (chunk_en / max(1, len(part_words))) * 0.45)
            else:
                chunk_lang = self.last_confirmed_language
                chunk_conf = 0.5

            segments.append(SegmentLanguage(text=part_str, language=chunk_lang, confidence=chunk_conf))

        for w in words:
            if w in _ROMAN_URDU_LEXICON:
                roman_urdu_matches += 1
                total_recognized += 1
            elif w in _ENGLISH_LEXICON:
                english_matches += 1
                total_recognized += 1

        total_words = len(words)
        ur_ratio = roman_urdu_matches / max(1, total_words)
        en_ratio = english_matches / max(1, total_words)

        # 3. Code-switching & Language Decision
        is_code_switched = (
            (roman_urdu_matches >= 2 or ur_ratio >= 0.20)
            and (english_matches >= 2 or en_ratio >= 0.20)
        )

        if ur_ratio >= 0.35 and en_ratio >= 0.20:
            # Strong code-switching (e.g. "yaar mujhe ye file today submit karni hai")
            detected_lang = "en_ur_mixed"
            conf = min(0.96, (ur_ratio + en_ratio) * 0.9)
            self.last_confirmed_language = "ur_roman"  # Treat as Roman-Urdu context
        elif roman_urdu_matches > english_matches or ur_ratio >= 0.25:
            detected_lang = "ur_roman"
            conf = min(0.95, 0.55 + (ur_ratio * 0.4))
            self.last_confirmed_language = "ur_roman"
        elif english_matches > roman_urdu_matches or en_ratio >= 0.25:
            detected_lang = "en"
            conf = min(0.95, 0.55 + (en_ratio * 0.4))
            self.last_confirmed_language = "en"
        else:
            # Low confidence / ambiguous — fallback to last confirmed language
            detected_lang = self.last_confirmed_language
            conf = 0.50

        # Script preference
        output_script = "latin"
        if detected_lang == "ur" or pref == "urdu_script":
            output_script = "arabic"

        return LanguageDetectionResult(
            language=detected_lang,
            confidence=round(conf, 3),
            is_code_switched=is_code_switched,
            output_script=output_script,
            segments=segments,
        )


# Global singleton instance
language_detector = LanguageDetector()
