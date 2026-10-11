"""test_local_engine_v3.py — Verification of VoxFlow v3 100% Local Modules.

Tests:
1. Silero VAD gating & silence duration accumulation.
2. Local ASR engine initialization & vocabulary biasing.
3. Local Cleanup Service (self-corrections, sequential actions, zero cloud calls).
4. VAD silence-duration implicit commit & trailing conjunction extension.
"""

import numpy as np
import pytest

from asr_service import asr_engine, LocalASREngine
from intent_engine import intent_engine, IntentAction
from local_cleanup_service import local_cleanup_service, llm_refine
from vad_service import vad_service, SileroVADService


def test_vad_service_silence_detection():
    vad = SileroVADService(threshold=0.5)
    vad.reset()

    # Pure silence (zeros)
    silence_chunk = np.zeros(1600, dtype=np.float32)
    res1 = vad.process_chunk(silence_chunk)
    assert not res1.is_speech
    assert res1.speech_probability < 0.20
    assert res1.silence_duration_ms >= 0.0


def test_vad_service_synthetic_speech_detection():
    vad = SileroVADService(threshold=0.3)
    vad.reset()

    # Synthetic voice tone (200Hz sine wave at normal conversational volume ~ -15dBFS)
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    speech_chunk = (0.25 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    res = vad.process_chunk(speech_chunk)
    assert res.is_speech
    assert res.speech_probability > 0.30
    assert res.speech_duration_ms >= 0.0


def test_asr_engine_biasing_and_gating():
    engine = LocalASREngine(model_size="tiny")
    engine.set_personal_vocabulary(["Kubernetes", "PyTorch", "VoxFlow"])
    assert "VoxFlow" in engine._personal_vocabulary

    # Empty audio
    empty_audio = np.array([], dtype=np.float32)
    res_empty = engine.transcribe_audio_array(empty_audio)
    assert res_empty.text == ""

    # Silence audio should be gated by VAD without crashing
    silence = np.zeros(3200, dtype=np.float32)
    res_silence = engine.transcribe_audio_array(silence, gate_with_vad=True)
    assert res_silence.gated_by_vad is True
    assert res_silence.text == ""


@pytest.mark.asyncio
async def test_local_cleanup_service_restructuring():
    # Sequential task structuring
    raw_tasks = "First thing we gotta do is fix the build error in CI, then after that ping Sarah, and lastly deploy to production."
    res = await local_cleanup_service.refine(raw_tasks)
    assert "Action items:" in res.refined_text
    assert "1. Fix the build error in CI" in res.refined_text
    assert "2. Ping Sarah" in res.refined_text
    assert "3. Deploy to production" in res.refined_text
    assert res.intent_score >= 0.50
    assert not res.drift_detected


@pytest.mark.asyncio
async def test_local_cleanup_service_self_correction():
    raw = "Let's schedule the meeting for Monday no wait Tuesday at 3 PM."
    res = await local_cleanup_service.refine(raw)
    assert "Tuesday" in res.refined_text
    assert "no wait" not in res.refined_text.lower()


@pytest.mark.asyncio
async def test_local_cleanup_llm_refine_drop_in():
    raw = "um basically I kinda wanna check the API response time"
    refined = await llm_refine(raw)
    assert "um" not in refined.lower()
    assert "basically" not in refined.lower()
    assert "want to" in refined.lower() or "check" in refined.lower()


def test_intent_engine_vad_pause_implicit_commit():
    # Case 1: Complete thought + short pause (300ms) -> PASS_THROUGH (user still actively speaking)
    res1 = intent_engine.parse_utterance("I am submitting the report now", silence_duration_ms=300.0)
    assert res1.action == IntentAction.PASS_THROUGH

    # Case 2: Complete thought + VAD silence duration >= 1200ms -> COMMIT
    res2 = intent_engine.parse_utterance("I am submitting the report now", silence_duration_ms=1300.0)
    assert res2.action == IntentAction.COMMIT
    assert res2.intended_text == "I am submitting the report now"

    # Case 3: Incomplete trailing thought ("and...") + 1300ms pause -> PENDING (extends wait)
    res3 = intent_engine.parse_utterance("I want to go to the store and", silence_duration_ms=1300.0)
    assert res3.action == IntentAction.PENDING
    assert not res3.is_complete

    # Case 4: Incomplete trailing thought ("and...") + 2500ms extended pause -> COMMIT (user stopped speaking)
    res4 = intent_engine.parse_utterance("I want to go to the store and", silence_duration_ms=2500.0)
    assert res4.action == IntentAction.COMMIT
    assert res4.intended_text == "I want to go to the store"
