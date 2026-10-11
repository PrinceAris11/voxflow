"""refiner_service.py — Service layer orchestrating speech refinement.

Coordinates:
- Language Identification (English, Urdu, Roman Urdu, Code-Switching)
- Adaptive memory filler filtering and vocabulary tracking
- Context-aware ML cleanup (comparison-safe 'like', self-correction, stutter collapsing)
- Dedicated punctuation restoration with question intonation
- Contextual emotion & emoji insertion (toggleable, default OFF)
- LRU caching with sub-millisecond lookups
- On-device deep refinement (self-correction resolution, action-item structuring)
- Asynchronous persistence
"""

from __future__ import annotations

import logging
import re
import time
from typing import Any

from adaptive_memory import AdaptiveMemory, SEED_FILLERS
from cache import transcript_cache
from cleanup_engine import cleanup_engine
from config import settings
from emoji_service import emoji_service
from language_detector import language_detector, LanguageDetectionResult
from intent_engine import intent_engine, IntentAction
from punctuation_service import punctuation_service, PunctuationStyle
from schemas import TranscriptResponse, MemorySummaryResponse
from local_cleanup_service import local_cleanup_service, llm_refine

log = logging.getLogger(__name__)

# Stop words excluded from best_word calculation
_STOP_WORDS = frozenset({
    "the", "and", "that", "this", "with", "have", "been", "from", "into",
    "your", "about", "for", "our", "will", "would", "should", "could", "can",
    "then", "them", "there", "they", "their", "more", "was", "are", "but",
    "not", "its", "has", "had", "were", "what", "when", "who", "how",
}) | SEED_FILLERS


class RefinerService:
    def __init__(self, memory: AdaptiveMemory | None = None):
        self.memory = memory or AdaptiveMemory(storage_path=settings.storage_path)

    @staticmethod
    def clean_transcript(text: str) -> str:
        """Baseline cleanup: context-aware normalization, contractions, and filler removal."""
        cleaned_res = cleanup_engine.clean_finalized(text, language="en")
        text = cleaned_res.cleaned_text
        text = re.sub(r"\b(?:i am|i'm)\b", "I am", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(?:thats|that's)\b", "that's", text, flags=re.IGNORECASE)
        text = re.sub(r"\b(?:dont|don't)\b", "don't", text, flags=re.IGNORECASE)
        text = re.sub(r"\bok\b", "okay", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            text = text[0].upper() + text[1:]
        return text

    @staticmethod
    def best_word(text: str) -> str:
        """Extract the most prominent content keyword."""
        words = re.findall(r"[a-zA-Z']+", text.lower())
        scored: dict[str, float] = {}
        for word in words:
            if len(word) <= 2 or word in _STOP_WORDS:
                continue
            scored[word] = scored.get(word, 0.0) + len(word) * 0.5 + 1.0
        if not scored:
            return "voice"
        return max(scored, key=lambda w: scored[w])

    async def refine(
        self,
        raw_text: str,
        use_llm: bool = True,
        record_memory: bool = True,
        language: str | None = None,
        punctuation_style: PunctuationStyle = "standard",
        enable_emoji: bool = False,
        emoji_theme: str | None = "standard",
        emoji_density: str | None = "balanced",
        output_script: str | None = None,
        silence_duration_ms: float = 0.0,
    ) -> TranscriptResponse:
        """Execute full speech processing pipeline:

        raw text
          -> Language Detection (Part B)
          -> Adaptive Memory observation
          -> Context-aware Cleanup & Self-Correction (Part C)
          -> Local Refinement / Structuring (Part C)
          -> Punctuation Restoration (Part D)
          -> Emoji Insertion (Part E, if enabled)
          -> Caching & Persistence
        """
        start_time = time.perf_counter()
        raw_clean = raw_text.strip()

        # Check Cache
        cache_key = (
            f"{raw_clean}::llm={use_llm}::lang={language}::punc={punctuation_style}::emoji={enable_emoji}::theme={emoji_theme}::density={emoji_density}"
        )
        cached_result = transcript_cache.get(cache_key)
        if cached_result:
            duration_ms = (time.perf_counter() - start_time) * 1000
            res = cached_result.model_copy()
            res.cached = True
            res.duration_ms = round(duration_ms, 2)
            log.debug("Cache hit for: %s", raw_clean[:30])
            return res

        # Step 1: Language Identification (Part B)
        if language:
            lid_result = LanguageDetectionResult(
                language=language,
                confidence=1.0,
                is_code_switched=False,
                output_script="latin" if language != "ur" else "arabic",
                segments=[],
            )
        else:
            lid_result = language_detector.detect(
                raw_clean,
                output_script_pref=output_script,
            )

        # Step 2: Intent Finalization & Smart Listener (Part E)
        intent_res = intent_engine.parse_utterance(
            raw_clean,
            language=lid_result.language,
            silence_duration_ms=silence_duration_ms,
        )
        if intent_res.action == IntentAction.DISCARD:
            mem_summary = self.memory.summary()
            duration_ms = (time.perf_counter() - start_time) * 1000
            return TranscriptResponse(
                raw=raw_clean,
                text_baseline="",
                text="",
                best_word="cancelled",
                memory=MemorySummaryResponse(**mem_summary),
                cached=False,
                duration_ms=round(duration_ms, 2),
                detected_language=lid_result.language,
                language_confidence=lid_result.confidence,
                is_code_switched=lid_result.is_code_switched,
                emoji_inserted=None,
                emotion_detected=None,
                valence="neutral",
                intent_score=1.0,
                drift_detected=False,
                intent_action=intent_res.action.value,
                discarded_text=intent_res.discarded_text,
                is_complete_intent=True,
            )

        working_input = intent_res.intended_text if intent_res.intended_text else raw_clean

        # Step 3: Adaptive Memory Observation & Adaptation
        if record_memory:
            self.memory.observe(working_input)

        adapted = self.memory.adapt_transcript(working_input)

        # Step 4: Context-Aware ML Cleanup & Self-Correction (Part C)
        cleanup_res = cleanup_engine.clean_finalized(
            adapted,
            language=lid_result.language,
        )
        baseline_cleaned = cleanup_res.cleaned_text

        # Step 5: Deep LLM Refinement (if enabled and configured)
        if use_llm:
            refined_candidate = await llm_refine(baseline_cleaned)
        else:
            refined_candidate = baseline_cleaned

        # Step 6: Dedicated Punctuation Restoration (Part D)
        punctuated_text = punctuation_service.punctuate(
            refined_candidate,
            language=lid_result.language,
            style=punctuation_style,
        )

        # Step 7: Emotion & Context-Aware Emoji Insertion (Part E, Toggleable)
        emoji_res = emoji_service.process(
            punctuated_text,
            enabled=enable_emoji,
            language=lid_result.language,
            theme=emoji_theme,
            density=emoji_density,
        )
        final_text = emoji_res.final_text

        # Prominent content word & memory snapshot
        prominent_word = self.best_word(final_text) if final_text else "empty"
        mem_summary = self.memory.summary()

        duration_ms = (time.perf_counter() - start_time) * 1000
        response = TranscriptResponse(
            raw=raw_clean,
            text_baseline=baseline_cleaned,
            text=final_text,
            best_word=prominent_word,
            memory=MemorySummaryResponse(**mem_summary),
            cached=False,
            duration_ms=round(duration_ms, 2),
            detected_language=lid_result.language,
            language_confidence=lid_result.confidence,
            is_code_switched=lid_result.is_code_switched,
            emoji_inserted=emoji_res.emoji_inserted,
            emotion_detected=emoji_res.detected_emotion,
            valence=emoji_res.valence,
            intent_score=cleanup_res.intent_preservation_score,
            drift_detected=cleanup_res.drift_detected,
            intent_action=intent_res.action.value,
            discarded_text=intent_res.discarded_text,
            is_complete_intent=intent_res.is_complete,
        )

        # Cache the result
        transcript_cache.set(cache_key, response)

        # Async non-blocking memory persistence
        if record_memory:
            await self.memory.async_save()

        return response

    async def process_job(self, job: Any) -> TranscriptResponse:
        """Handler for JobManager background worker execution."""
        return await self.refine(
            raw_text=job.raw_text,
            use_llm=job.use_llm,
            record_memory=True,
            language=getattr(job, "language", None),
            punctuation_style=getattr(job, "punctuation_style", "standard"),
            enable_emoji=getattr(job, "enable_emoji", False),
        )


# Global refiner service singleton
refiner_service = RefinerService()
