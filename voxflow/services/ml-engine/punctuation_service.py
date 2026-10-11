"""punctuation_service.py — Explicit Punctuation Restoration Engine.

Runs as a dedicated stage AFTER language identification and cleanup.
Restores:
- Sentence boundaries and terminals (. ? !)
- Question marks from phrasing (wh-words, auxiliary verbs, and casual question tags like 'right?', 'hai na?')
- Commas at natural transitional boundaries ('however', 'therefore', 'lekin', 'phir')
- Sentence-initial and proper-noun capitalization (APIs, tools, acronyms)
- Configurable styles: 'casual', 'standard', 'formal'
"""

from __future__ import annotations

import re
from typing import Literal

PunctuationStyle = Literal["casual", "standard", "formal"]

# Question starters (English)
_EN_WH_WORDS = re.compile(
    r"^(?:what|why|how|when|where|who|whom|which|whose)\b",
    re.IGNORECASE,
)
_EN_INVERTED_AUX = re.compile(
    r"^(?:is|are|was|were|can|could|should|would|will|do|does|did|have|has|had|am|may|might)\s+(?:i|you|he|she|it|we|they|this|that|there)\b",
    re.IGNORECASE,
)

# Tag questions in English & Roman Urdu (casual speech: "you're coming right" -> "You're coming, right?")
_TAG_QUESTION_ENDINGS = re.compile(
    r"\b(?:right|correct|you\s+know|isn't\s+it|aren't\s+you|don't\s+you|hai\s+na|na|kya)\s*$",
    re.IGNORECASE,
)

# Urdu / Roman Urdu question indicators
_URDU_QUESTION_WORDS = re.compile(
    r"\b(?:kya|kyun|kyu|kab|kahan|kaise|kis\s+tarah|konsa|konsi)\b",
    re.IGNORECASE,
)

# Transitional adverbs that take a comma
_TRANSITIONAL_CONNECTORS = [
    r"\bhowever\b", r"\btherefore\b", r"\bmoreover\b", r"\bfurthermore\b",
    r"\bmeanwhile\b", r"\bin\s+fact\b", r"\bon\s+the\s+other\s+hand\b",
    r"\blekin\b", r"\bphir\s+bhi\b", r"\bwaise\b",
]
_COMPILED_CONNECTORS = re.compile(f"^(?:{'|'.join(_TRANSITIONAL_CONNECTORS)})", re.IGNORECASE)

# Technical acronyms and brand names to properly capitalize
_TECH_ACRONYMS: dict[str, str] = {
    "api": "API",
    "apis": "APIs",
    "ci": "CI",
    "cd": "CD",
    "ui": "UI",
    "ux": "UX",
    "db": "DB",
    "sql": "SQL",
    "nosql": "NoSQL",
    "redis": "Redis",
    "docker": "Docker",
    "figma": "Figma",
    "aws": "AWS",
    "gcp": "GCP",
    "rest": "REST",
    "http": "HTTP",
    "https": "HTTPS",
    "json": "JSON",
    "xml": "XML",
    "html": "HTML",
    "css": "CSS",
    "js": "JS",
    "ts": "TS",
    "python": "Python",
    "voxflow": "VoxFlow",
    "github": "GitHub",
}


class PunctuationService:
    """Restores punctuation, casing, and structure based on speech prosody cues and language patterns."""

    def punctuate(
        self,
        text: str,
        language: str = "en",
        style: PunctuationStyle = "standard",
    ) -> str:
        """Punctuate cleaned text according to language rules and style preference."""
        if not text or not text.strip():
            return ""

        text = text.strip()

        # Step 1: Normalize spacing around existing punctuation
        text = re.sub(r"\s+([,.;:?!])", r"\1", text)
        text = re.sub(r"([,.;:?!])(?=[^\s\d])", r"\1 ", text)

        # Step 2: Split into sentences or clauses if run-on
        sentences = self._split_clauses(text)
        punctuated_sentences: list[str] = []

        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue

            # Capitalize first letter
            s_clean = s_clean[0].upper() + s_clean[1:]

            # Technical acronym & entity capitalization
            s_clean = self._capitalize_technical_terms(s_clean)

            # Insert natural commas at transitional connectors
            if style in ("standard", "formal"):
                s_clean = self._add_transitional_commas(s_clean)

            # Determine sentence terminal: Question (?), Exclamation (!), or Period (.)
            terminal = self._determine_terminal(s_clean, language, style)

            # Handle tag question commas: e.g. "You're coming, right?"
            s_clean = self._format_tag_question(s_clean, terminal)

            # Strip any trailing comma or dangling punctuation before appending terminal
            s_clean = re.sub(r"[,;:\s]+$", "", s_clean)

            if not s_clean.endswith((".", "!", "?")):
                if style == "casual" and len(s_clean.split()) <= 4 and terminal == ".":
                    # In casual style, short commands/phrases leave terminal off for conversational feel
                    pass
                else:
                    s_clean += terminal

            s_clean = re.sub(r",\s*([.?!])", r"\1", s_clean)
            s_clean = re.sub(r"\s+([,.;:?!])", r"\1", s_clean)
            punctuated_sentences.append(s_clean)

        result = " ".join(punctuated_sentences)

        # Step 3: Paragraph break on clear multi-thought shifts
        if style in ("standard", "formal") and len(punctuated_sentences) >= 4:
            # Group into 2-sentence paragraphs
            paragraphs: list[str] = []
            for i in range(0, len(punctuated_sentences), 2):
                paragraphs.append(" ".join(punctuated_sentences[i:i + 2]))
            result = "\n\n".join(paragraphs)

        return result

    def _split_clauses(self, text: str) -> list[str]:
        """Split text into sentences while respecting existing punctuation."""
        parts = re.split(r"(?<=[.?!])\s+", text)
        if len(parts) > 1:
            return parts

        # If completely unpunctuated run-on, split at strong conjunctions
        # e.g., "we need to fix the server and also we should cache the endpoint"
        split_pattern = re.compile(
            r"\s+(?=(?:and\s+also|furthermore|moreover|lekin|aur\s+phir)\b)",
            re.IGNORECASE,
        )
        return split_pattern.split(text)

    def _determine_terminal(
        self,
        sentence: str,
        language: str,
        style: PunctuationStyle,
    ) -> str:
        """Analyze if sentence is a question, exclamation, or statement."""
        raw = sentence.rstrip(".,;:?!")

        # Exclamations
        if re.search(r"\b(?:awesome|kamaal|zabardast|amazing|wow|hurray|congrats|great job)\b", raw, re.IGNORECASE):
            return "!"

        # Question check 1: Wh-question starter
        if _EN_WH_WORDS.search(raw):
            return "?"

        # Question check 2: Inverted auxiliary verb starter
        if _EN_INVERTED_AUX.search(raw):
            return "?"

        # Question check 3: Greetings followed by interrogative clause ("hi how are you", "hey can you check this")
        if re.search(r"^(?:hi|hello|hey|salam)\b.*?\b(?:how|what|where|when|who|why|kaisa|kaise|kya|are\s+you|can\s+you)\b", raw, re.IGNORECASE):
            return "?"


        # Question check 3: Question tag ending ("you're coming right", "tum aa rahe ho na")
        if _TAG_QUESTION_ENDINGS.search(raw):
            return "?"

        # Question check 4: Urdu question marker
        if language in ("ur", "ur_roman", "en_ur_mixed") and _URDU_QUESTION_WORDS.search(raw):
            return "?"

        return "."

    def _format_tag_question(self, sentence: str, terminal: str) -> str:
        """Place comma before trailing question tag: 'You're coming, right?'"""
        if terminal != "?":
            return sentence

        match = _TAG_QUESTION_ENDINGS.search(sentence.rstrip(".,;:?!"))
        if match and match.start() > 2:
            tag = match.group(0).strip()
            prefix = sentence[:match.start()].rstrip(" ,;")
            return f"{prefix}, {tag}"
        return sentence

    def _add_transitional_commas(self, sentence: str) -> str:
        """Add comma after leading transitional words: 'However we should...' -> 'However, we should...'"""
        words = sentence.split(maxsplit=1)
        if not words:
            return sentence

        first_word = words[0].rstrip(",;:")
        if first_word.lower() in ("however", "therefore", "moreover", "meanwhile", "lekin", "hi", "hello", "hey", "salam"):
            rest = words[1] if len(words) > 1 else ""
            if rest and not rest.startswith(","):
                return f"{first_word}, {rest}"
        return sentence

    def _capitalize_technical_terms(self, text: str) -> str:
        """Properly capitalize tech stacks, acronyms, and tools."""
        def repl(match: re.Match) -> str:
            w = match.group(0).lower()
            return _TECH_ACRONYMS.get(w, match.group(0))

        pattern = rf"\b(?:{'|'.join(_TECH_ACRONYMS.keys())})\b"
        return re.sub(pattern, repl, text, flags=re.IGNORECASE)


# Global singleton punctuation service
punctuation_service = PunctuationService()
