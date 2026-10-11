"""interactive_test.py — Interactive tester for VoxFlow Silero VAD Service.

Allows you to test VAD using:
1. Live microphone input (using standard Linux arecord).
2. Synthetic audio simulations (silence, normal speech, whispered tones).
3. Custom WAV audio file input.
4. Latency benchmark runner.

Usage:
    python interactive_test.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from vad_service import SileroVADService, vad_service

__test__ = False


def test_microphone():
    """Record 3 seconds from system microphone and analyze through VAD."""
    print("\n--- [1] LIVE MICROPHONE TEST ---")
    print("Testing for audio recording tools...")
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name

    try:
        print("\n🎤 RECORDING IN 3 SECONDS... GET READY TO SPEAK!")
        for sec in range(3, 0, -1):
            print(f"   {sec}...")
            time.sleep(1)

        print("🎙️ RECORDING NOW (Speak for 3 seconds into your mic)...")
        # 3 seconds, 16kHz mono 16-bit
        cmd = ["arecord", "-d", "3", "-r", "16000", "-c", "1", "-f", "S16_LE", tmp_wav]
        subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)
        print("✓ Recording finished! Processing through Silero VAD...\n")

        # Read audio
        audio, sr = sf.read(tmp_wav, dtype="float32")
        vad = SileroVADService()
        vad.reset()

        chunk_size = 1600  # 100ms chunks
        total_chunks = len(audio) // chunk_size

        print(f"{'Time':<10} | {'Probability':<12} | {'Speech?':<8} | {'Silence Duration':<18} | Visual Energy")
        print("-" * 75)

        for i in range(total_chunks):
            chunk = audio[i * chunk_size : (i + 1) * chunk_size]
            t_s = (i * 100) / 1000.0
            res = vad.process_chunk(chunk)
            bar_len = int(res.speech_probability * 25)
            bar = "█" * bar_len + "░" * (25 - bar_len)
            status = "SPEECH" if res.is_speech else "SILENCE"
            print(f"{t_s:4.1f}s - {t_s+0.1:4.1f}s | {res.speech_probability:^10.4f} | {status:^8} | {res.silence_duration_ms:8.1f} ms    | [{bar}]")

        print("-" * 75)
        print(f"Backend Used : {vad.backend.upper()} (spec: {vad.model_version})")
        print("✓ Microphone test complete!\n")

    except FileNotFoundError:
        print("❌ 'arecord' utility not found. Please test using synthetic signals or custom wav file.")
    except Exception as exc:
        print(f"❌ Recording error: {exc}")
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


def test_synthetic_signals():
    """Simulate audio signals and inspect VAD outputs."""
    print("\n--- [2] SYNTHETIC SIGNALS TEST ---")
    vad = SileroVADService()
    vad.reset()

    # 1. Silence
    silence = np.zeros(1600, dtype=np.float32)
    res_sil = vad.process_chunk(silence)
    print(f"• Pure Silence (100ms)      : Speech={res_sil.is_speech:<5} | Prob={res_sil.speech_probability:.4f} | Silence={res_sil.silence_duration_ms}ms")

    # 2. Voice Tone (200Hz conversational volume)
    t = np.linspace(0, 0.1, 1600, endpoint=False)
    voice = (0.25 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    res_voice = vad.process_chunk(voice)
    print(f"• Conversational Tone (100ms): Speech={res_voice.is_speech:<5} | Prob={res_voice.speech_probability:.4f} | SpeechStarted={res_voice.speech_started}")

    # 3. Soft Whisper (300Hz at -35dBFS)
    whisper = (0.02 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    res_whisper = vad.process_chunk(whisper)
    print(f"• Quiet Signal (100ms)      : Speech={res_whisper.is_speech:<5} | Prob={res_whisper.speech_probability:.4f}")
    print("✓ Synthetic signal test complete!\n")


def test_file_playback():
    """Test with a custom WAV file."""
    print("\n--- [3] CUSTOM WAV FILE TEST ---")
    filepath = input("Enter path to a 16kHz WAV file: ").strip()
    p = Path(filepath)
    if not p.exists():
        print(f"❌ File not found: {filepath}")
        return

    audio, sr = sf.read(str(p), dtype="float32")
    if sr != 16000:
        print(f"⚠️ Note: Audio sample rate is {sr}Hz (Silero VAD expects 16000Hz).")

    vad = SileroVADService()
    vad.reset()

    chunk_size = int(sr * 0.1)  # 100ms
    speech_chunks = 0
    total_chunks = len(audio) // chunk_size

    for i in range(total_chunks):
        chunk = audio[i * chunk_size : (i + 1) * chunk_size]
        res = vad.process_chunk(chunk)
        if res.is_speech:
            speech_chunks += 1

    ratio = (speech_chunks / max(1, total_chunks)) * 100.0
    print(f"✓ Analyzed {total_chunks} chunks ({total_chunks * 0.1:.1f}s)")
    print(f"  Speech Detected in : {speech_chunks}/{total_chunks} chunks ({ratio:.1f}%)")


def main():
    while True:
        print("=" * 60)
        print("   VOXFLOW VAD SERVICE — INTERACTIVE TEST MENU")
        print("=" * 60)
        print("1. Test with Live Microphone (records 3 seconds)")
        print("2. Test with Synthetic Audio Signals")
        print("3. Test with a WAV Audio File")
        print("4. Run Latency Benchmark Harness")
        print("5. Run Full Pytest Suite")
        print("6. Exit")
        print("=" * 60)

        choice = input("Enter your choice (1-6): ").strip()

        if choice == "1":
            test_microphone()
        elif choice == "2":
            test_synthetic_signals()
        elif choice == "3":
            test_file_playback()
        elif choice == "4":
            subprocess.run([sys.executable, "vad_benchmark.py", "--iterations", "100"])
        elif choice == "5":
            subprocess.run([sys.executable, "-m", "pytest", "tests/test_vad_service.py", "tests/test_vad_benchmark.py", "-v"])
        elif choice == "6":
            print("Exiting. Have a great day!")
            break
        else:
            print("Invalid choice, please enter 1, 2, 3, 4, 5, or 6.\n")


if __name__ == "__main__":
    main()
