"""asr_service.py — Local On-Device Speech Recognition Engine for VoxFlow.

Features:
1. 100% on-device local ASR via faster-whisper (CTranslate2 backend).
2. CPU-quantized execution (int8) optimized for laptop/desktop hardware.
3. Multilingual automatic language identification with code-switching support (Urdu & English).
4. Personal vocabulary biasing via initial_prompt (supporting Part H personalization).
5. VAD-gated chunk transcription to eliminate wasteful inference during silence.
"""

from __future__ import annotations

import io
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from vad_service import vad_service

log = logging.getLogger(__name__)


@dataclass
class ASRSegment:
    """A single transcribed segment with timing and language metrics."""
    text: str
    start: float
    end: float
    language: str
    probability: float
    no_speech_prob: float


@dataclass
class ASRTranscriptionResult:
    """Result of an on-device speech-to-text pass."""
    text: str
    detected_language: str
    language_probability: float
    segments: list[ASRSegment] = field(default_factory=list)
    duration_ms: float = 0.0
    is_code_switched: bool = False
    gated_by_vad: bool = False


class LocalASREngine:
    """Local Speech-to-Text engine using faster-whisper CTranslate2 quantized models."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        download_root: str | Path | None = None,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.download_root = str(download_root) if download_root else None

        self._model: Any = None
        self._last_confirmed_language: str = "en"
        self._personal_vocabulary: list[str] = []

    def set_personal_vocabulary(self, words: list[str]) -> None:
        """Update personal vocabulary list for ASR decoder prompt biasing."""
        self._personal_vocabulary = [w.strip() for w in words if w.strip()]

    def _get_model(self) -> Any:
        """Lazy initialization of faster-whisper model."""
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
            log.info(
                "Loading local faster-whisper model '%s' (device=%s, compute=%s)...",
                self.model_size,
                self.device,
                self.compute_type,
            )
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=self.download_root,
                cpu_threads=4,
            )
            log.info("Local faster-whisper model successfully loaded.")
            return self._model
        except Exception as exc:
            log.warning("Faster-whisper model initialization failed or deferred (%s).", exc)
            return None

    def transcribe_audio_array(
        self,
        audio_array: np.ndarray,
        language: str | None = None,
        initial_prompt: str | None = None,
        gate_with_vad: bool = True,
    ) -> ASRTranscriptionResult:
        """Transcribe a 16kHz float32 numpy audio array on-device.
        
        Args:
            audio_array: 1D float32 audio samples (16,000 Hz).
            language: Optional language code ('en', 'ur', etc.). If None, auto-detected.
            initial_prompt: Optional biasing prompt for domain terms and personal vocabulary.
            gate_with_vad: If True, tests audio with VAD first to avoid inference on silence.
        """
        start_time = time.perf_counter()

        if len(audio_array) == 0:
            return ASRTranscriptionResult(
                text="",
                detected_language=self._last_confirmed_language,
                language_probability=1.0,
                duration_ms=0.0,
            )

        # Step 1: VAD Gating
        if gate_with_vad:
            vad_res = vad_service.process_chunk(audio_array)
            if not vad_res.is_speech and vad_res.speech_probability < 0.20:
                elapsed = (time.perf_counter() - start_time) * 1000.0
                return ASRTranscriptionResult(
                    text="",
                    detected_language=self._last_confirmed_language,
                    language_probability=1.0,
                    duration_ms=round(elapsed, 2),
                    gated_by_vad=True,
                )

        # Step 2: Prepare decoder prompt biasing
        prompt_parts: list[str] = []
        if self._personal_vocabulary:
            prompt_parts.append(", ".join(self._personal_vocabulary))
        if initial_prompt:
            prompt_parts.append(initial_prompt)
        combined_prompt = "; ".join(prompt_parts) if prompt_parts else None

        # Step 3: Run local ASR inference
        model = self._get_model()
        if model is None:
            # Fallback if model weights are not loaded or in mock test environment
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return ASRTranscriptionResult(
                text="",
                detected_language=self._last_confirmed_language,
                language_probability=1.0,
                duration_ms=round(elapsed, 2),
            )

        try:
            segments_gen, info = model.transcribe(
                audio_array,
                language=language,
                beam_size=3,
                best_of=3,
                temperature=0.0,
                condition_on_previous_text=False,
                initial_prompt=combined_prompt,
                vad_filter=False,  # We gate in front with Silero VAD
            )

            # Language ID Handling & Low-Confidence Fallback
            det_lang = info.language if info and info.language else self._last_confirmed_language
            det_prob = info.language_probability if info else 1.0

            if det_prob < 0.40:
                det_lang = self._last_confirmed_language
            else:
                self._last_confirmed_language = det_lang

            segments: list[ASRSegment] = []
            text_pieces: list[str] = []
            distinct_languages: set[str] = {det_lang}

            for seg in segments_gen:
                clean_seg_text = seg.text.strip()
                if clean_seg_text:
                    text_pieces.append(clean_seg_text)
                    segments.append(
                        ASRSegment(
                            text=clean_seg_text,
                            start=seg.start,
                            end=seg.end,
                            language=det_lang,
                            probability=seg.avg_logprob,
                            no_speech_prob=seg.no_speech_prob,
                        )
                    )

            full_text = " ".join(text_pieces).strip()
            is_code_switched = len(distinct_languages) > 1 or (
                det_lang in ("ur", "en") and any(w in full_text.lower() for w in ["the", "is", "hai", "karo", "meeting"])
            )

            elapsed = (time.perf_counter() - start_time) * 1000.0
            return ASRTranscriptionResult(
                text=full_text,
                detected_language=det_lang,
                language_probability=round(det_prob, 3),
                segments=segments,
                duration_ms=round(elapsed, 2),
                is_code_switched=is_code_switched,
                gated_by_vad=False,
            )

        except Exception as exc:
            log.error("Local ASR transcription error: %s", exc)
            elapsed = (time.perf_counter() - start_time) * 1000.0
            return ASRTranscriptionResult(
                text="",
                detected_language=self._last_confirmed_language,
                language_probability=0.0,
                duration_ms=round(elapsed, 2),
            )


# Global singleton ASR instance
asr_engine = LocalASREngine()
