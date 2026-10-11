"""test_vad_service.py — Comprehensive unit and integration tests for SileroVADService.

Covers:
1. ONNX neural model execution and acoustic fallback mode.
2. Multiple input formats: int16 PCM bytes, float32 ndarray, int16 ndarray, stereo downmixing.
3. Edge cases: empty chunks, odd-length byte buffers, sub-window and multi-window chunks, NaNs/Infs.
4. State machine dynamics: speech_started, speech_ended, debounced micro-pauses, silence accumulation.
5. Thread-safe execution under concurrent calls.
"""

import threading
import numpy as np
import pytest

from vad_service import SileroVADService, VADResult, vad_service


@pytest.fixture
def onnx_vad() -> SileroVADService:
    vad = SileroVADService(mode="onnx", threshold=0.4)
    vad.reset()
    return vad


@pytest.fixture
def acoustic_vad() -> SileroVADService:
    vad = SileroVADService(mode="acoustic", threshold=0.4)
    vad.reset()
    return vad


# ---------------------------------------------------------------------------
# 1. Backend & Model Initialization Tests
# ---------------------------------------------------------------------------

def test_vad_service_singleton():
    assert vad_service is not None
    assert isinstance(vad_service, SileroVADService)
    assert vad_service.sample_rate == 16000


def test_onnx_initialization(onnx_vad):
    if onnx_vad.is_onnx_available:
        assert onnx_vad.backend == "onnx"
        assert onnx_vad.model_version in ("v6", "v5", "v4")
    else:
        assert onnx_vad.backend == "acoustic"


def test_acoustic_fallback(acoustic_vad):
    assert acoustic_vad.backend == "acoustic"
    assert not acoustic_vad.is_onnx_available


# ---------------------------------------------------------------------------
# 2. Audio Format & Conversion Tests
# ---------------------------------------------------------------------------

def test_empty_audio_chunks(onnx_vad):
    # Empty bytes
    res_bytes = onnx_vad.process_chunk(b"")
    assert not res_bytes.is_speech
    assert res_bytes.speech_probability == 0.0
    assert res_bytes.samples_processed == 0

    # Empty numpy array
    res_arr = onnx_vad.process_chunk(np.zeros(0, dtype=np.float32))
    assert not res_arr.is_speech
    assert res_arr.speech_probability == 0.0
    assert res_arr.samples_processed == 0

    # Odd-length bytes (e.g. 3 bytes instead of even 16-bit word)
    res_odd = onnx_vad.process_chunk(b"\x00\x01\x02")
    assert res_odd.samples_processed == 1


def test_pcm_bytes_vs_ndarray(acoustic_vad):
    # Generate 512 samples of a 250Hz tone
    t = np.linspace(0, 512 / 16000, 512, endpoint=False)
    float_samples = (0.3 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
    int16_samples = (float_samples * 32767).astype(np.int16)
    pcm_bytes = int16_samples.tobytes()

    res_arr = acoustic_vad.process_chunk(float_samples)
    acoustic_vad.reset()
    res_bytes = acoustic_vad.process_chunk(pcm_bytes)

    assert abs(res_arr.speech_probability - res_bytes.speech_probability) < 0.05
    assert res_arr.is_speech == res_bytes.is_speech


def test_stereo_downmix(acoustic_vad):
    # 512 samples of stereo audio (shape: 512, 2)
    t = np.linspace(0, 512 / 16000, 512, endpoint=False)
    channel1 = (0.25 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    channel2 = (0.25 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    stereo = np.column_stack([channel1, channel2])

    res = acoustic_vad.process_chunk(stereo)
    assert res.samples_processed == 512
    assert res.is_speech


def test_nan_and_inf_handling(acoustic_vad):
    # Buffer with NaNs and Infs must be sanitized without raising exceptions
    corrupt = np.array([np.nan, np.inf, -np.inf, 0.5, 0.2], dtype=np.float32)
    res = acoustic_vad.process_chunk(corrupt)
    assert isinstance(res.speech_probability, float)
    assert not np.isnan(res.speech_probability)


# ---------------------------------------------------------------------------
# 3. Speech & Silence Detection Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["acoustic", "onnx"])
def test_silence_detection(mode):
    vad = SileroVADService(mode=mode, threshold=0.4)
    vad.reset()

    # 100ms of pure silence (1600 zeros)
    silence = np.zeros(1600, dtype=np.float32)
    res = vad.process_chunk(silence)

    assert not res.is_speech
    assert res.is_silence
    assert res.speech_probability < 0.20
    assert res.silence_duration_ms == pytest.approx(100.0, abs=1.0)
    assert res.speech_duration_ms == 0.0


@pytest.mark.parametrize("mode", ["acoustic", "onnx"])
def test_speech_detection(mode):
    vad = SileroVADService(mode=mode, threshold=0.3)
    vad.reset()

    # Synthetic harmonic tone at conversational level (~ -15 dBFS)
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    speech = (0.25 * np.sin(2 * np.pi * 200 * t) + 0.10 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)

    res = vad.process_chunk(speech)

    assert res.is_speech
    assert res.speech_probability >= 0.30
    assert res.speech_started is True
    assert res.speech_duration_ms == pytest.approx(100.0, abs=1.0)
    assert res.silence_duration_ms == 0.0


# ---------------------------------------------------------------------------
# 4. State Machine & Duration Accumulation Tests
# ---------------------------------------------------------------------------

def test_conversational_state_machine(acoustic_vad):
    """Simulate a realistic sequence: silence -> speech burst -> micro-pause -> speech -> extended silence."""
    acoustic_vad.reset()

    # 1. 200ms initial silence (2 chunks of 100ms)
    silence_chunk = np.zeros(1600, dtype=np.float32)
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    speech_chunk = (0.25 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)

    res1 = acoustic_vad.process_chunk(silence_chunk)
    assert not res1.is_speech
    assert not res1.speech_started
    assert res1.silence_duration_ms == pytest.approx(100.0, abs=1.0)

    res2 = acoustic_vad.process_chunk(silence_chunk)
    assert res2.silence_duration_ms == pytest.approx(200.0, abs=1.0)

    # 2. 300ms speech burst (3 chunks of 100ms)
    res3 = acoustic_vad.process_chunk(speech_chunk)
    assert res3.is_speech
    assert res3.speech_started is True  # Pulse onset
    assert res3.speech_ended is False
    assert res3.speech_duration_ms == pytest.approx(100.0, abs=1.0)
    assert res3.silence_duration_ms == 0.0

    res4 = acoustic_vad.process_chunk(speech_chunk)
    assert res4.is_speech
    assert res4.speech_started is False  # Sustained, no re-trigger
    assert res4.speech_duration_ms == pytest.approx(200.0, abs=1.0)

    res5 = acoustic_vad.process_chunk(speech_chunk)
    assert res5.speech_duration_ms == pytest.approx(300.0, abs=1.0)

    # 3. 100ms micro-pause (1 chunk of silence, less than 300ms speech_pad_ms)
    res6 = acoustic_vad.process_chunk(silence_chunk)
    assert not res6.is_speech
    assert res6.speech_started is False
    assert res6.speech_ended is False  # Must NOT declare speech ended during a 100ms micro-pause
    assert res6.speech_duration_ms == pytest.approx(300.0, abs=1.0)
    assert res6.silence_duration_ms == pytest.approx(100.0, abs=1.0)

    # 4. Speech resumes for 200ms
    res7 = acoustic_vad.process_chunk(speech_chunk)
    assert res7.is_speech
    assert res7.speech_started is False  # Already in speech session
    assert res7.speech_duration_ms == pytest.approx(400.0, abs=1.0)

    res8 = acoustic_vad.process_chunk(speech_chunk)
    assert res8.speech_duration_ms == pytest.approx(500.0, abs=1.0)

    # 5. Extended silence: 4 chunks (400ms > 300ms speech_pad_ms)
    res9 = acoustic_vad.process_chunk(silence_chunk)  # 100ms
    assert not res9.speech_ended

    res10 = acoustic_vad.process_chunk(silence_chunk)  # 200ms
    assert not res10.speech_ended

    res11 = acoustic_vad.process_chunk(silence_chunk)  # 300ms -> speech_pad_ms reached!
    assert res11.speech_ended is True  # Exactly on threshold

    res12 = acoustic_vad.process_chunk(silence_chunk)  # 400ms
    assert res12.speech_ended is False  # Edge trigger cleared
    assert res12.silence_duration_ms == pytest.approx(400.0, abs=1.0)

    # 6. Silence accumulates until Part E implicit commit threshold (1200ms)
    for _ in range(8):  # +800ms = 1200ms total
        res_final = acoustic_vad.process_chunk(silence_chunk)

    assert res_final.silence_duration_ms >= 1200.0


# ---------------------------------------------------------------------------
# 5. Reset & Dict Serialization Tests
# ---------------------------------------------------------------------------

def test_vad_reset(acoustic_vad):
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    speech = (0.25 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)

    res = acoustic_vad.process_chunk(speech)
    assert res.speech_duration_ms > 0

    acoustic_vad.reset()
    assert acoustic_vad._accumulated_speech_ms == 0.0
    assert acoustic_vad._accumulated_silence_ms == 0.0
    assert not acoustic_vad._is_active_speech


def test_vad_result_serialization():
    res = VADResult(
        is_speech=True,
        speech_probability=0.88765,
        silence_duration_ms=0.0,
        speech_duration_ms=350.2,
        speech_started=True,
        speech_ended=False,
        backend="onnx",
        raw_probability=0.8999,
        samples_processed=1600,
        audio_position_ms=350.2,
    )
    d = res.to_dict()
    assert d["is_speech"] is True
    assert d["speech_probability"] == 0.8877
    assert d["speech_duration_ms"] == 350.2
    assert d["backend"] == "onnx"
    assert d["samples_processed"] == 1600


# ---------------------------------------------------------------------------
# 6. Concurrency / Thread Safety Tests
# ---------------------------------------------------------------------------

def test_concurrent_vad_processing(acoustic_vad):
    chunk = np.zeros(512, dtype=np.float32)
    errors = []

    def worker():
        try:
            for _ in range(50):
                res = acoustic_vad.process_chunk(chunk)
                assert isinstance(res, VADResult)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent VAD execution failed with errors: {errors}"
