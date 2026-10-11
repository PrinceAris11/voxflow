"""test_vad_benchmark.py — Automated verification of VAD latency benchmarks and evaluation harness."""

import pytest
from vad_benchmark import VADBenchmarkHarness
from vad_service import SileroVADService


@pytest.fixture
def harness() -> VADBenchmarkHarness:
    return VADBenchmarkHarness(sample_rate=16000)


def test_vad_latency_sla_compliance(harness):
    """Ensure VAD meets strict real-time latency budgets (p95 < 4ms for 32ms frame, RTF < 0.05)."""
    vad = SileroVADService(mode="auto")

    # Benchmark 32ms frame (512 samples)
    metrics_32ms = harness.benchmark_chunk(
        vad=vad,
        chunk_samples=512,
        iterations=100,
        warmup=20,
        input_format="pcm_bytes",
        signal_type="speech",
        sla_p95_target_ms=4.0,
    )
    assert metrics_32ms.sla_passed is True
    assert metrics_32ms.p95_ms < 4.0
    assert metrics_32ms.rtf < 0.10
    assert metrics_32ms.speedup_multiplier > 10.0

    # Benchmark 100ms chunk (1600 samples)
    metrics_100ms = harness.benchmark_chunk(
        vad=vad,
        chunk_samples=1600,
        iterations=100,
        warmup=20,
        input_format="float32",
        signal_type="speech",
        sla_p95_target_ms=8.0,
    )
    assert metrics_100ms.sla_passed is True
    assert metrics_100ms.p95_ms < 8.0
    assert metrics_100ms.rtf < 0.05


def test_streaming_conversational_evaluation(harness):
    """Test full conversational streaming evaluation harness."""
    report = harness.eval_streaming_session(backend="auto", chunk_size_ms=100.0)

    assert report.all_events_accurate is True
    assert report.implicit_commit_triggered is True
    assert report.speech_started_events >= 1
    assert report.speech_ended_events >= 1
    assert 0.30 <= report.gating_ratio <= 0.45
    assert report.mean_latency_ms < 5.0
    assert report.p95_latency_ms < 8.0
    assert report.rtf < 0.05


def test_acoustic_fallback_sla(harness):
    """Verify that pure acoustic fallback is ultra-fast (p95 < 1.0ms, RTF < 0.01)."""
    vad_acoustic = SileroVADService(mode="acoustic")

    metrics = harness.benchmark_chunk(
        vad=vad_acoustic,
        chunk_samples=1600,  # 100ms
        iterations=100,
        warmup=20,
        input_format="pcm_bytes",
        signal_type="speech",
        sla_p95_target_ms=1.5,
    )
    assert metrics.sla_passed is True
    assert metrics.p95_ms < 1.0
    assert metrics.rtf < 0.005
    assert metrics.speedup_multiplier > 200.0
