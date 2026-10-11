"""vad_benchmark.py — Latency Evaluation Harness and Performance Benchmark for VoxFlow VAD.

Measures:
1. Per-chunk latency across frame sizes (16ms, 32ms, 64ms, 100ms, 250ms, 500ms).
2. Latency percentiles (min, p50, p90, p95, p99, max, jitter/stddev).
3. Real-Time Factor (RTF) and speedup multiplier (Nx real-time).
4. Raw 16-bit PCM bytes decoding vs Float32 NumPy input overhead.
5. End-to-end streaming conversational session evaluation with silence-gating compute savings.
6. SLA budget compliance checks and exportable JSON/Markdown reports.

Usage:
    python vad_benchmark.py [--iterations 300] [--backend all|onnx|acoustic] [--json report.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from vad_service import SileroVADService, VADResult


@dataclass
class LatencyMetrics:
    """Latency distribution and throughput statistics for a benchmark scenario."""

    scenario: str
    backend: str
    chunk_samples: int
    chunk_duration_ms: float
    iterations: int
    input_format: str
    signal_type: str

    min_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    mean_ms: float
    std_ms: float

    rtf: float  # Real-Time Factor: mean_latency_ms / chunk_duration_ms
    speedup_multiplier: float  # chunk_duration_ms / mean_latency_ms
    throughput_chunks_sec: float
    throughput_audio_sec_per_sec: float
    sla_passed: bool  # True if p95 meets budget target

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StreamingSessionReport:
    """Results from evaluating an end-to-end streaming conversational session."""

    session_duration_s: float
    chunks_processed: int
    chunk_size_ms: float
    backend: str
    speech_duration_s: float
    silence_duration_s: float
    gating_ratio: float  # Fraction of audio where transcription can be skipped
    speech_started_events: int
    speech_ended_events: int
    implicit_commit_triggered: bool
    mean_latency_ms: float
    p95_latency_ms: float
    max_latency_ms: float
    rtf: float
    all_events_accurate: bool


class VADBenchmarkHarness:
    """Benchmark and latency evaluation harness for VoxFlow Voice Activity Detection."""

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def generate_synthetic_signal(
        self,
        signal_type: Literal["silence", "noise", "speech"],
        num_samples: int,
    ) -> np.ndarray:
        """Generate test audio vectors for evaluation."""
        if signal_type == "silence":
            return np.zeros(num_samples, dtype=np.float32)

        if signal_type == "noise":
            # White/Gaussian ambient room noise at -45 dBFS
            rms_target = 10.0 ** (-45.0 / 20.0)  # ~0.0056
            noise = np.random.normal(0.0, rms_target, num_samples).astype(np.float32)
            return np.clip(noise, -1.0, 1.0)

        if signal_type == "speech":
            # Harmonic multi-formant synthesized speech tone (~ -15 dBFS)
            t = np.linspace(0, num_samples / self.sample_rate, num_samples, endpoint=False)
            f0 = 180.0  # Fundamental frequency typical of human voice
            signal = (
                0.20 * np.sin(2 * np.pi * f0 * t)
                + 0.12 * np.sin(2 * np.pi * 2 * f0 * t)
                + 0.08 * np.sin(2 * np.pi * 3 * f0 * t)
                + 0.04 * np.sin(2 * np.pi * 5 * f0 * t)
            ).astype(np.float32)
            return signal

        return np.zeros(num_samples, dtype=np.float32)

    def benchmark_chunk(
        self,
        vad: SileroVADService,
        chunk_samples: int,
        iterations: int = 300,
        warmup: int = 50,
        input_format: Literal["float32", "pcm_bytes"] = "float32",
        signal_type: Literal["silence", "noise", "speech"] = "speech",
        sla_p95_target_ms: float = 5.0,
    ) -> LatencyMetrics:
        """Execute calibrated latency benchmark on a specific chunk configuration."""
        audio_float = self.generate_synthetic_signal(signal_type, chunk_samples)

        if input_format == "pcm_bytes":
            int16_samples = (audio_float * 32767).astype(np.int16)
            test_payload = int16_samples.tobytes()
        else:
            test_payload = audio_float

        chunk_duration_ms = (chunk_samples / self.sample_rate) * 1000.0

        # Warm-up phase to prime caches and memory
        vad.reset()
        for _ in range(warmup):
            vad.process_chunk(test_payload)

        # Measured execution
        latencies_ms: list[float] = []
        vad.reset()

        for _ in range(iterations):
            t0 = time.perf_counter()
            vad.process_chunk(test_payload)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        latencies = np.array(latencies_ms, dtype=np.float64)
        mean_ms = float(np.mean(latencies))
        min_ms = float(np.min(latencies))
        max_ms = float(np.max(latencies))
        std_ms = float(np.std(latencies))
        p50_ms = float(np.percentile(latencies, 50))
        p90_ms = float(np.percentile(latencies, 90))
        p95_ms = float(np.percentile(latencies, 95))
        p99_ms = float(np.percentile(latencies, 99))

        rtf = mean_ms / max(0.001, chunk_duration_ms)
        speedup = chunk_duration_ms / max(1e-6, mean_ms)
        throughput_chunks = 1000.0 / max(1e-6, mean_ms)
        throughput_audio = speedup

        scenario_name = f"{chunk_duration_ms:.0f}ms ({chunk_samples} samples) - {input_format} [{signal_type}]"

        return LatencyMetrics(
            scenario=scenario_name,
            backend=vad.backend,
            chunk_samples=chunk_samples,
            chunk_duration_ms=chunk_duration_ms,
            iterations=iterations,
            input_format=input_format,
            signal_type=signal_type,
            min_ms=round(min_ms, 4),
            p50_ms=round(p50_ms, 4),
            p90_ms=round(p90_ms, 4),
            p95_ms=round(p95_ms, 4),
            p99_ms=round(p99_ms, 4),
            max_ms=round(max_ms, 4),
            mean_ms=round(mean_ms, 4),
            std_ms=round(std_ms, 4),
            rtf=round(rtf, 5),
            speedup_multiplier=round(speedup, 1),
            throughput_chunks_sec=round(throughput_chunks, 1),
            throughput_audio_sec_per_sec=round(throughput_audio, 1),
            sla_passed=p95_ms <= sla_p95_target_ms,
        )

    def run_suite(
        self,
        backends: list[str] | None = None,
        iterations: int = 250,
    ) -> list[LatencyMetrics]:
        """Run full latency matrix across chunk sizes and backends."""
        if backends is None:
            backends = ["acoustic"]
            # Test if ONNX is available
            test_vad = SileroVADService(mode="onnx")
            if test_vad.is_onnx_available:
                backends.append("onnx")

        # Standard real-time streaming audio chunk configurations
        chunk_configs = [
            (256, 3.0),   # 16ms  (budget: 3ms)
            (512, 4.0),   # 32ms  (budget: 4ms)
            (1024, 6.0),  # 64ms  (budget: 6ms)
            (1600, 8.0),  # 100ms (budget: 8ms)
            (4000, 15.0), # 250ms (budget: 15ms)
            (8000, 25.0), # 500ms (budget: 25ms)
        ]

        results: list[LatencyMetrics] = []

        for b_name in backends:
            vad = SileroVADService(mode=b_name)  # type: ignore[arg-type]

            for chunk_size, budget in chunk_configs:
                # 1. Float32 array benchmark
                res_float = self.benchmark_chunk(
                    vad=vad,
                    chunk_samples=chunk_size,
                    iterations=iterations,
                    input_format="float32",
                    signal_type="speech",
                    sla_p95_target_ms=budget,
                )
                results.append(res_float)

                # 2. Raw PCM bytes benchmark (tests end-to-end buffer decoding)
                res_bytes = self.benchmark_chunk(
                    vad=vad,
                    chunk_samples=chunk_size,
                    iterations=iterations,
                    input_format="pcm_bytes",
                    signal_type="speech",
                    sla_p95_target_ms=budget,
                )
                results.append(res_bytes)

        return results

    def eval_streaming_session(
        self,
        backend: Literal["auto", "onnx", "acoustic"] = "auto",
        chunk_size_ms: float = 100.0,
    ) -> StreamingSessionReport:
        """Simulate and evaluate an end-to-end 10-second dictation session.

        Timeline:
          - 0.0s to 1.0s: Ambient room silence (1000ms)
          - 1.0s to 4.5s: First utterance ("Hey VoxFlow, open dashboard", 3500ms)
          - 4.5s to 4.8s: Natural inter-word micro-pause (300ms)
          - 4.8s to 7.5s: Second utterance ("and show today's stats", 2700ms)
          - 7.5s to 10.0s: Trailing silence leading to implicit commit (2500ms)
        """
        vad = SileroVADService(mode=backend, min_silence_duration_ms=1200.0)
        vad.reset()

        chunk_samples = int((chunk_size_ms / 1000.0) * self.sample_rate)
        total_chunks = int(10000 / chunk_size_ms)  # 100 chunks for 10.0s

        silence_chunk = self.generate_synthetic_signal("silence", chunk_samples)
        speech_chunk = self.generate_synthetic_signal("speech", chunk_samples)

        latencies_ms: list[float] = []
        speech_started_count = 0
        speech_ended_count = 0
        gated_chunks = 0
        implicit_commit_reached = False

        for i in range(total_chunks):
            current_time_s = (i * chunk_size_ms) / 1000.0

            # Determine signal for this timeslot
            if 0.0 <= current_time_s < 1.0:
                audio = silence_chunk
                expected_speech = False
            elif 1.0 <= current_time_s < 4.5:
                audio = speech_chunk
                expected_speech = True
            elif 4.5 <= current_time_s < 4.8:
                audio = silence_chunk
                expected_speech = False
            elif 4.8 <= current_time_s < 7.5:
                audio = speech_chunk
                expected_speech = True
            else:
                audio = silence_chunk
                expected_speech = False

            t0 = time.perf_counter()
            res = vad.process_chunk(audio)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

            if res.speech_started:
                speech_started_count += 1
            if res.speech_ended:
                speech_ended_count += 1
            if not res.is_speech and res.speech_probability < 0.20:
                gated_chunks += 1
            if res.silence_duration_ms >= 1200.0:
                implicit_commit_reached = True

        lat = np.array(latencies_ms)
        mean_lat = float(np.mean(lat))
        p95_lat = float(np.percentile(lat, 95))
        max_lat = float(np.max(lat))
        rtf = mean_lat / chunk_size_ms

        gating_ratio = gated_chunks / float(total_chunks)
        # Expected: speech_started occurred on first utterance and possibly after hesitation
        # implicit commit was achieved during trailing silence
        all_accurate = (
            speech_started_count >= 1
            and speech_ended_count >= 1
            and implicit_commit_reached
            and gating_ratio >= 0.25
        )

        return StreamingSessionReport(
            session_duration_s=10.0,
            chunks_processed=total_chunks,
            chunk_size_ms=chunk_size_ms,
            backend=vad.backend,
            speech_duration_s=6.2,
            silence_duration_s=3.8,
            gating_ratio=round(gating_ratio, 3),
            speech_started_events=speech_started_count,
            speech_ended_events=speech_ended_count,
            implicit_commit_triggered=implicit_commit_reached,
            mean_latency_ms=round(mean_lat, 4),
            p95_latency_ms=round(p95_lat, 4),
            max_latency_ms=round(max_lat, 4),
            rtf=round(rtf, 5),
            all_events_accurate=all_accurate,
        )


def format_table(results: list[LatencyMetrics]) -> str:
    """Format benchmark results into a clean ASCII table."""
    headers = [
        "Scenario",
        "Backend",
        "Dur(ms)",
        "Mean(ms)",
        "p50(ms)",
        "p95(ms)",
        "p99(ms)",
        "Max(ms)",
        "RTF",
        "Speedup",
        "SLA",
    ]

    rows: list[list[str]] = []
    for r in results:
        sla_mark = "PASS" if r.sla_passed else "FAIL"
        rows.append([
            r.scenario[:32],
            r.backend,
            f"{r.chunk_duration_ms:.0f}",
            f"{r.mean_ms:.3f}",
            f"{r.p50_ms:.3f}",
            f"{r.p95_ms:.3f}",
            f"{r.p99_ms:.3f}",
            f"{r.max_ms:.3f}",
            f"{r.rtf:.4f}",
            f"{r.speedup_multiplier:.0f}x",
            sla_mark,
        ])

    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    separator = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_str = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"

    lines = [separator, header_str, separator]
    for row in rows:
        line = "| " + " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
        lines.append(line)
    lines.append(separator)

    return "\n".join(lines)


def format_markdown(results: list[LatencyMetrics], session: StreamingSessionReport) -> str:
    """Generate Markdown report suitable for GitHub / documentation."""
    lines = [
        "# VoxFlow VAD Latency Benchmark & Evaluation Report",
        "",
        "## 1. Executive Summary",
        f"- **VAD Engine**: Silero VAD (ONNX Runtime + Acoustic Fallback)",
        f"- **Evaluation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Primary Backend Evaluated**: `{session.backend}`",
        f"- **Streaming RTF (Real-Time Factor)**: `{session.rtf:.5f}` ({(1.0 / session.rtf):.0f}x faster than real-time)",
        f"- **Silence Gating Savings**: `{session.gating_ratio * 100:.1f}%` of audio segments bypassed transcription inference",
        f"- **Implicit Commit Triggered**: `{session.implicit_commit_triggered}` (silence duration >= 1200ms)",
        f"- **State Machine Accuracy**: `{'VERIFIED' if session.all_events_accurate else 'FAILED'}`",
        "",
        "## 2. Latency Distribution Matrix",
        "",
        "| Scenario | Backend | Chunk (ms) | Mean (ms) | p50 (ms) | p95 (ms) | Max (ms) | RTF | Speedup | SLA |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        sla_badge = "✅ PASS" if r.sla_passed else "❌ FAIL"
        lines.append(
            f"| {r.scenario} | `{r.backend}` | {r.chunk_duration_ms:.0f} | "
            f"{r.mean_ms:.3f} | {r.p50_ms:.3f} | {r.p95_ms:.3f} | {r.max_ms:.3f} | "
            f"`{r.rtf:.4f}` | **{r.speedup_multiplier:.0f}x** | {sla_badge} |"
        )

    lines.extend([
        "",
        "## 3. Streaming Conversational Session Evaluation",
        f"- **Total Session Duration**: `{session.session_duration_s:.1f}s`",
        f"- **Speech Duration**: `{session.speech_duration_s:.1f}s`",
        f"- **Silence Duration**: `{session.silence_duration_s:.1f}s`",
        f"- **Speech Onset Pulses**: `{session.speech_started_events}`",
        f"- **Speech End Transitions**: `{session.speech_ended_events}`",
        f"- **Streaming Mean Latency**: `{session.mean_latency_ms:.3f}ms`",
        f"- **Streaming p95 Latency**: `{session.p95_latency_ms:.3f}ms`",
        f"- **Streaming Max Latency**: `{session.max_latency_ms:.3f}ms`",
        "",
    ])

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxFlow VAD Latency Benchmark & Eval Harness")
    parser.add_argument("--iterations", type=int, default=200, help="Benchmark iterations per scenario")
    parser.add_argument("--backend", choices=["all", "onnx", "acoustic"], default="all", help="Backend to evaluate")
    parser.add_argument("--json", type=str, default="", help="Path to write JSON benchmark report")
    parser.add_argument("--markdown", type=str, default="", help="Path to write Markdown benchmark report")
    args = parser.parse_args()

    harness = VADBenchmarkHarness()

    backends = None
    if args.backend != "all":
        backends = [args.backend]

    print("=" * 80)
    print("VoxFlow VAD Latency Benchmark & Evaluation Harness")
    print("=" * 80)
    print(f"Running calibrated benchmark (iterations={args.iterations})...\n")

    results = harness.run_suite(backends=backends, iterations=args.iterations)
    print(format_table(results))

    print("\nRunning streaming conversational session evaluation...")
    session_report = harness.eval_streaming_session()
    print(f"  Session Duration     : {session_report.session_duration_s}s")
    print(f"  Backend Used         : {session_report.backend}")
    print(f"  Silence Gating Ratio : {session_report.gating_ratio * 100:.1f}%")
    print(f"  Mean Latency         : {session_report.mean_latency_ms:.3f}ms")
    print(f"  p95 Latency          : {session_report.p95_latency_ms:.3f}ms")
    print(f"  Implicit Commit      : {'SUCCESS' if session_report.implicit_commit_triggered else 'FAIL'}")
    print(f"  Events Accuracy      : {'VERIFIED' if session_report.all_events_accurate else 'FAILED'}")

    # Output exports
    if args.json:
        out_path = Path(args.json)
        out_dict = {
            "metrics": [r.to_dict() for r in results],
            "streaming_session": asdict(session_report),
        }
        out_path.write_text(json.dumps(out_dict, indent=2))
        print(f"\n[+] JSON report saved to: {out_path.resolve()}")

    if args.markdown:
        md_path = Path(args.markdown)
        md_content = format_markdown(results, session_report)
        md_path.write_text(md_content)
        print(f"[+] Markdown report saved to: {md_path.resolve()}")


if __name__ == "__main__":
    main()
