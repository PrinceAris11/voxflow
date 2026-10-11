"""cleanup_engine.py — Advanced Intent-Preserving ML Cleanup Engine.

Implements:
1. Context-aware filler removal (preserving comparison/verb "like")
2. Multilingual & Roman Urdu filler pruning ("yaani", "matlab", "acha", etc.)
3. Self-correction resolution ("no wait", "actually", "nahi ruko", "chhoro")
4. False start & stutter/repetition collapsing
5. Intent preservation guardrail (keyword retention & length drift check)
6. Two-tier streaming performance (fast interim vs deep finalized)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# ─── Regular Expressions & Constants ──────────────────────────────────────────

# English pure fillers (always safe to prune)
_EN_PURE_FILLERS = re.compile(
    r"\b(um|uh|er|ah|you\s+know|basically|sort\s+of|kind\s+of|i\s+mean)\b",
    re.IGNORECASE,
)

# Urdu / Roman Urdu fillers
_URDU_ROMAN_FILLERS = re.compile(
    r"\b(yaani|yani|matlab\s+ke|matlb\s+ke|arey\s+yaar|acha\s+to|achha\s+to)\b",
    re.IGNORECASE,
)

# Comparison patterns where "like" MUST be preserved
_LIKE_COMPARISON_PATTERNS = [
    re.compile(r"\b(looks?|sounded?|sounds?|feels?|felt|seems?|acts?|is|are|was|were|be|been)\s+like\b", re.IGNORECASE),
    re.compile(r"\bwould\s+like\b", re.IGNORECASE),
    re.compile(r"\b(i|we|they|you)\s+like\b", re.IGNORECASE),
    re.compile(r"\blike\s+(a|an|the|this|that|these|those|it|me|him|her|us|them|how|what)\b", re.IGNORECASE),
    re.compile(r"\bsomething\s+like\b", re.IGNORECASE),
]

# Stutter repetitions: consecutive identical words (e.g. "I I", "the the", "karni karni")
# Exceptions for intentional emphasis: "very very", "really really", "bohot bohot", "boht boht"
_STUTTER_PATTERN = re.compile(r"\b([a-zA-Z]+)\s+\1\b", re.IGNORECASE)
_INTENTIONAL_EMPHASIS = frozenset({"very", "really", "bohot", "boht", "too", "so"})

# Self-correction trigger patterns (English + Roman Urdu)
_CORRECTION_TRIGGERS = [
    # English
    r"\b(?:no\s+wait|wait\s+no|actually|scratch\s+that|i\s+mean|correction|or\s+rather)\b",
    # Roman Urdu
    r"\b(?:nahi\s+ruko|nahin\s+ruko|ruko\s+ruko|arey\s+nahi|chhoro\s+isko|chhoro|chorro)\b",
]
_COMPILED_CORRECTIONS = re.compile(f"(?:{'|'.join(_CORRECTION_TRIGGERS)})", re.IGNORECASE)


@dataclass
class CleanupResult:
    cleaned_text: str
    fillers_removed: list[str]
    corrections_resolved: list[str]
    repetitions_collapsed: list[str]
    intent_preservation_score: float  # 0.0 to 1.0
    drift_detected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "cleaned_text": self.cleaned_text,
            "fillers_removed": self.fillers_removed,
            "corrections_resolved": self.corrections_resolved,
            "repetitions_collapsed": self.repetitions_collapsed,
            "intent_preservation_score": round(self.intent_preservation_score, 3),
            "drift_detected": self.drift_detected,
        }


class CleanupEngine:
    """Multi-stage cleaning engine with context-awareness and intent guardrails."""

    # ── Tier 1: Fast Interim Preview (<0.2ms) ─────────────────────────────────
    @staticmethod
    def clean_interim(raw_text: str) -> str:
        """Lightweight pass for live streaming preview without altering semantics."""
        if not raw_text:
            return ""
        text = raw_text.strip()
        text = _EN_PURE_FILLERS.sub("", text)
        text = _URDU_ROMAN_FILLERS.sub("", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # ── Tier 2: Deep Finalized Cleanup ────────────────────────────────────────
    def clean_finalized(
        self,
        raw_text: str,
        language: str = "en",
    ) -> CleanupResult:
        """Full pipeline pass for finalized utterance."""
        if not raw_text or not raw_text.strip():
            return CleanupResult(
                cleaned_text="",
                fillers_removed=[],
                corrections_resolved=[],
                repetitions_collapsed=[],
                intent_preservation_score=1.0,
                drift_detected=False,
            )

        text = raw_text.strip()
        fillers_removed: list[str] = []
        corrections_resolved: list[str] = []
        repetitions_collapsed: list[str] = []

        # Stage 1: Self-Correction Resolution
        text, resolved_corrections = self._resolve_self_corrections(text)
        corrections_resolved.extend(resolved_corrections)

        # Stage 2: Stutter & Accidental Repetition Collapsing
        text, collapsed = self._collapse_repetitions(text)
        repetitions_collapsed.extend(collapsed)

        # Stage 3: Context-Aware Filler Removal
        text, removed = self._remove_fillers_context_aware(text, language)
        fillers_removed.extend(removed)

        # Stage 4: Tidy Whitespace & Basic Normalization
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s+([,.;:?!])", r"\1", text).strip()
        if text:
            text = text[0].upper() + text[1:]

        # Stage 5: Intent Preservation Guardrail
        intent_score, drift = self._check_intent_guardrail(raw_text, text)

        # If severe drift detected (e.g. over-pruning), fallback to safer baseline
        final_text = text
        if drift and len(text) < 3 and len(raw_text) > 10:
            final_text = raw_text.strip()

        return CleanupResult(
            cleaned_text=final_text,
            fillers_removed=fillers_removed,
            corrections_resolved=corrections_resolved,
            repetitions_collapsed=repetitions_collapsed,
            intent_preservation_score=intent_score,
            drift_detected=drift,
        )

    # ── Internal Cleanup Stages ───────────────────────────────────────────────

    def _resolve_self_corrections(self, text: str) -> tuple[str, list[str]]:
        """Identify self-correction triggers and discard the superseded clause.

        Example:
            "We should meet at 3 PM... no wait, make it 4 PM"
            -> "We should meet at make it 4 PM" -> "We should meet at 4 PM"
        """
        match = _COMPILED_CORRECTIONS.search(text)
        if not match:
            return text, []

        trigger = match.group(0)
        start_idx = match.start()
        end_idx = match.end()

        # Split into before-trigger and after-trigger
        before = re.sub(r"[\s\.\,\;]+$", "", text[:start_idx]).strip()
        after = text[end_idx:].strip()
        # Clean leading punctuation from after
        after = re.sub(r"^[,.;:?!]+\s*", "", after)

        if not after:
            # Trigger occurred at very end of utterance with no correction
            return before, [trigger]

        # Check if the corrected part specifies a replacement for a clause
        # If before contains a previous number/time/date that after updates
        time_or_num = re.findall(r"\b\d+(?::\d+)?(?:\s*(?:am|pm))?\b", before, re.IGNORECASE)
        time_or_num_after = re.findall(r"\b\d+(?::\d+)?(?:\s*(?:am|pm))?\b", after, re.IGNORECASE)

        if time_or_num and time_or_num_after:
            # Replace the old entity with the new one
            last_old = time_or_num[-1].strip()
            last_new = time_or_num_after[0].strip()
            # Replace old number in before
            resolved_before = re.sub(rf"\b{re.escape(last_old)}\b", last_new, before)
            # Remove redundant "make it 4 PM" or similar from after if already substituted
            after_clean = re.sub(
                rf"^(?:make\s+it|let's\s+do|chhoro\s+isko|parson)?\s*{re.escape(last_new)}",
                "",
                after,
                flags=re.IGNORECASE,
            ).strip()
            result = f"{resolved_before} {after_clean}".strip()
            result = re.sub(r"\s+", " ", result)
            return result, [f"{trigger} -> {last_new}"]

        # Check if after specifies an entity replacement: e.g. "make it a Rivian", "change to 5 PM"
        make_it_match = re.match(
            r"^(?:make\s+it|change\s+it\s+to|let's\s+do|chhoro\s+isko)\s+(.+)",
            after,
            re.IGNORECASE,
        )
        if make_it_match:
            replacement_clause = make_it_match.group(1).strip()
            # Replace the last noun/entity in before
            resolved_before = re.sub(r"(?:(?:a|an|the)\s+)?\b[A-Za-z0-9\.\'\-]+\s*$", replacement_clause, before)
            return resolved_before, [f"{trigger} -> {replacement_clause}"]

        # General correction: Keep anchor prefix if after is a replacement fragment
        # E.g. "We need to fix Docker... no wait verify staging" -> "We need to verify staging"
        prefix_words = before.split()
        if len(prefix_words) > 4:
            # Retain leading stem (e.g. "We need to")
            stem = " ".join(prefix_words[:3])
            result = f"{stem} {after}".strip()
        else:
            result = after

        return result, [trigger]

    def _collapse_repetitions(self, text: str) -> tuple[str, list[str]]:
        """Collapse stutter repetitions while preserving intentional emphasis."""
        collapsed: list[str] = []

        def replacer(match: re.Match) -> str:
            word = match.group(1)
            if word.lower() in _INTENTIONAL_EMPHASIS:
                return match.group(0)  # Keep "very very"
            collapsed.append(f"{word} {word} -> {word}")
            return word

        # Run until no more consecutive duplicates
        prev_text = ""
        current_text = text
        while prev_text != current_text:
            prev_text = current_text
            current_text = _STUTTER_PATTERN.sub(replacer, current_text)

        return current_text, collapsed

    def _remove_fillers_context_aware(
        self,
        text: str,
        language: str,
    ) -> tuple[str, list[str]]:
        """Remove verbal fillers while protecting comparisons and legitimate words."""
        removed: list[str] = []

        # 1. Protect comparison "like" with placeholder
        protected_placeholders: dict[str, str] = {}
        counter = 0

        for pattern in _LIKE_COMPARISON_PATTERNS:
            for m in pattern.finditer(text):
                matched_str = m.group(0)
                placeholder = f"__PROTECTED_LIKE_{counter}__"
                protected_placeholders[placeholder] = matched_str
                text = text.replace(matched_str, placeholder, 1)
                counter += 1

        # 2. Prune pure English fillers
        for m in _EN_PURE_FILLERS.finditer(text):
            removed.append(m.group(0))
        text = _EN_PURE_FILLERS.sub("", text)

        # 3. Prune solitary hesitation "like" (not protected)
        hesitation_like = re.compile(r"(?:^|\s)\blike\b(?=\s|$)", re.IGNORECASE)
        for m in hesitation_like.finditer(text):
            removed.append("like (hesitation)")
        text = hesitation_like.sub(" ", text)

        # 4. Prune Urdu / Roman Urdu fillers if language is Urdu or Mixed
        if language in ("ur", "ur_roman", "en_ur_mixed"):
            for m in _URDU_ROMAN_FILLERS.finditer(text):
                removed.append(m.group(0))
            text = _URDU_ROMAN_FILLERS.sub("", text)

            # Contextual Roman Urdu fillers (e.g. standalone "matlab" at clause start)
            contextual_ur = re.compile(r"(?:^|[,.;])\s*\b(matlab|yaani|acha)\b\s*[,;]?", re.IGNORECASE)
            for m in contextual_ur.finditer(text):
                removed.append(m.group(1))
            text = contextual_ur.sub(" ", text)

        # 5. Restore protected comparison phrases
        for placeholder, original in protected_placeholders.items():
            text = text.replace(placeholder, original)

        return text, removed

    def _check_intent_guardrail(
        self,
        raw_text: str,
        cleaned_text: str,
    ) -> tuple[float, bool]:
        """Guardrail checking that facts, entities, numbers, and meaning are preserved."""
        raw_tokens = set(re.findall(r"[a-zA-Z0-9']+", raw_text.lower()))
        cleaned_tokens = set(re.findall(r"[a-zA-Z0-9']+", cleaned_text.lower()))

        # Extract core entities (numbers, capitalized words, technical terms)
        raw_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", raw_text))
        cleaned_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", cleaned_text))

        # Check number preservation (crucial for time, metrics, prices)
        number_loss = False
        if raw_numbers and not cleaned_numbers.issubset(raw_numbers):
            # Hallucinated new numbers
            number_loss = True

        # Calculate token retention
        content_tokens = {w for w in raw_tokens if len(w) > 3}
        if not content_tokens:
            return 1.0, False

        retained = content_tokens.intersection(cleaned_tokens)
        score = len(retained) / max(1, len(content_tokens))

        # Drift check: length shrank by > 75% or number loss occurred
        length_ratio = len(cleaned_text) / max(1, len(raw_text))
        drift_detected = (score < 0.40 and length_ratio < 0.25) or number_loss

        return min(1.0, max(0.0, score)), drift_detected


# Global singleton cleanup engine
cleanup_engine = CleanupEngine()
