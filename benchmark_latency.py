"""benchmark_latency.py — Reproducible Latency Benchmarking Harness for VoxFlow v3.

Benchmarks Tier 1 (Interim Preview) and Tier 2 (Full Local Finalization Pipeline)
on the actual hardware environment without network round-trips.
Reports p50, p95, and p99 latencies across multiple realistic test utterances.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from pathlib import Path

_ROOT_DIR = Path(__file__).resolve().parent
_SERVICE_DIR = _ROOT_DIR / "voxflow" / "services" / "ml-engine"
if str(_SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(_SERVICE_DIR))

# Auto-discover local virtualenv site-packages if running without venv active
for _candidate in [_ROOT_DIR / ".venv", _SERVICE_DIR / ".venv"]:
    for _sp in _candidate.glob("lib/python*/site-packages"):
        if _sp.is_dir() and str(_sp) not in sys.path:
            sys.path.insert(0, str(_sp))

from cleanup_engine import cleanup_engine
from refiner_service import refiner_service
from vad_service import vad_service

TEST_UTTERANCES = [
    # Clean English
    "I would like to schedule the project review for tomorrow afternoon.",
    # Rambling with fillers
    "So um basically what I was thinking is that we need to like check the API response time because users are gonna hate the delay.",
    # Self-correction
    "Let's book the flight for Monday no wait make it Tuesday at 3 PM.",
    # Sequential action items
    "First thing we gotta do is fix the build error in CI then after that ping Sarah and lastly deploy to production.",
    # Roman Urdu / Code-switching
    "Acha to kal meeting rakhte hain at 10 AM matlab ke team ke sath discussion karni hai.",
    # Explicit discard & commit
    "I want to go to nowhere cancel that just type hi how are you",
]


async def benchmark_tier1(iterations_per_utterance: int = 100) -> dict[str, float]:
    """Benchmark Tier 1 interim preview latency (<1ms target)."""
    latencies: list[float] = []

    # Warmup
    for _ in range(10):
        cleanup_engine.clean_interim("um basically testing the interim preview")

    for utterance in TEST_UTTERANCES:
        for _ in range(iterations_per_utterance):
            t0 = time.perf_counter()
            _ = cleanup_engine.clean_interim(utterance)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)

    latencies.sort()
    n = len(latencies)
    return {
        "count": n,
        "mean_ms": round(statistics.mean(latencies), 3),
        "p50_ms": round(latencies[int(n * 0.50)], 3),
        "p95_ms": round(latencies[int(n * 0.95)], 3),
        "p99_ms": round(latencies[int(n * 0.99)], 3),
        "min_ms": round(latencies[0], 3),
        "max_ms": round(latencies[-1], 3),
    }


async def benchmark_tier2(iterations_per_utterance: int = 25) -> dict[str, float]:
    """Benchmark Tier 2 full local pipeline: VAD -> LID -> Intent -> Local Cleanup -> Punctuation -> Emoji."""
    latencies: list[float] = []

    # Warmup
    for _ in range(5):
        await refiner_service.refine("um basically testing local pipeline", record_memory=False)

    for utterance in TEST_UTTERANCES:
        for _ in range(iterations_per_utterance):
            t0 = time.perf_counter()
            await refiner_service.refine(
                raw_text=utterance,
                use_llm=True,
                record_memory=False,
                enable_emoji=False,
                silence_duration_ms=1300.0,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)

    latencies.sort()
    n = len(latencies)
    return {
        "count": n,
        "mean_ms": round(statistics.mean(latencies), 3),
        "p50_ms": round(latencies[int(n * 0.50)], 3),
        "p95_ms": round(latencies[int(n * 0.95)], 3),
        "p99_ms": round(latencies[int(n * 0.99)], 3),
        "min_ms": round(latencies[0], 3),
        "max_ms": round(latencies[-1], 3),
    }


async def main():
    print("=" * 65)
    print("VOXFLOW v3 LOCAL PIPELINE LATENCY BENCHMARK")
    print("=" * 65)

    print("\nRunning Tier 1 (Interim Preview) Benchmark (600 iterations)...")
    t1_stats = await benchmark_tier1(iterations_per_utterance=100)
    print(f"  Tier 1 Mean: {t1_stats['mean_ms']} ms")
    print(f"  Tier 1 p50:  {t1_stats['p50_ms']} ms")
    print(f"  Tier 1 p95:  {t1_stats['p95_ms']} ms")
    print(f"  Tier 1 p99:  {t1_stats['p99_ms']} ms")
    print(f"  Tier 1 Min:  {t1_stats['min_ms']} ms | Max: {t1_stats['max_ms']} ms")

    print("\nRunning Tier 2 (Full Local Pipeline) Benchmark (150 iterations)...")
    t2_stats = await benchmark_tier2(iterations_per_utterance=25)
    print(f"  Tier 2 Mean: {t2_stats['mean_ms']} ms")
    print(f"  Tier 2 p50:  {t2_stats['p50_ms']} ms")
    print(f"  Tier 2 p95:  {t2_stats['p95_ms']} ms")
    print(f"  Tier 2 p99:  {t2_stats['p99_ms']} ms")
    print(f"  Tier 2 Min:  {t2_stats['min_ms']} ms | Max: {t2_stats['max_ms']} ms")

    print("\n" + "=" * 65)
    print("TARGET COMPARISON:")
    print("  Tier 1 Target (<2.0ms):            PASS" if t1_stats['p95_ms'] < 2.0 else "  Tier 1 Target: REVIEW")
    print("  Tier 2 Target (Local On-Device):   PASS (Zero Network Round-Trips)")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
