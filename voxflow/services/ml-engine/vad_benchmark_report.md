# VoxFlow VAD Latency Benchmark & Evaluation Report

## 1. Executive Summary
- **VAD Engine**: Silero VAD (ONNX Runtime + Acoustic Fallback)
- **Evaluation Date**: 2026-09-30 16:57:44
- **Primary Backend Evaluated**: `onnx`
- **Streaming RTF (Real-Time Factor)**: `0.00673` (149x faster than real-time)
- **Silence Gating Savings**: `38.0%` of audio segments bypassed transcription inference
- **Implicit Commit Triggered**: `True` (silence duration >= 1200ms)
- **State Machine Accuracy**: `VERIFIED`

## 2. Latency Distribution Matrix

| Scenario | Backend | Chunk (ms) | Mean (ms) | p50 (ms) | p95 (ms) | Max (ms) | RTF | Speedup | SLA |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 16ms (256 samples) - float32 [speech] | `acoustic` | 16 | 0.116 | 0.108 | 0.162 | 0.217 | `0.0072` | **138x** | ✅ PASS |
| 16ms (256 samples) - pcm_bytes [speech] | `acoustic` | 16 | 0.073 | 0.068 | 0.101 | 0.125 | `0.0046` | **219x** | ✅ PASS |
| 32ms (512 samples) - float32 [speech] | `acoustic` | 32 | 0.088 | 0.084 | 0.112 | 0.128 | `0.0027` | **363x** | ✅ PASS |
| 32ms (512 samples) - pcm_bytes [speech] | `acoustic` | 32 | 0.079 | 0.065 | 0.153 | 0.191 | `0.0025` | **407x** | ✅ PASS |
| 64ms (1024 samples) - float32 [speech] | `acoustic` | 64 | 0.109 | 0.093 | 0.183 | 0.263 | `0.0017` | **586x** | ✅ PASS |
| 64ms (1024 samples) - pcm_bytes [speech] | `acoustic` | 64 | 0.098 | 0.078 | 0.193 | 0.326 | `0.0015` | **655x** | ✅ PASS |
| 100ms (1600 samples) - float32 [speech] | `acoustic` | 100 | 0.132 | 0.113 | 0.210 | 0.314 | `0.0013` | **755x** | ✅ PASS |
| 100ms (1600 samples) - pcm_bytes [speech] | `acoustic` | 100 | 0.110 | 0.095 | 0.172 | 0.324 | `0.0011` | **914x** | ✅ PASS |
| 250ms (4000 samples) - float32 [speech] | `acoustic` | 250 | 0.136 | 0.113 | 0.223 | 0.513 | `0.0006` | **1834x** | ✅ PASS |
| 250ms (4000 samples) - pcm_bytes [speech] | `acoustic` | 250 | 0.097 | 0.087 | 0.138 | 0.176 | `0.0004` | **2578x** | ✅ PASS |
| 500ms (8000 samples) - float32 [speech] | `acoustic` | 500 | 0.183 | 0.166 | 0.290 | 0.347 | `0.0004` | **2730x** | ✅ PASS |
| 500ms (8000 samples) - pcm_bytes [speech] | `acoustic` | 500 | 0.096 | 0.081 | 0.144 | 0.169 | `0.0002` | **5234x** | ✅ PASS |
| 16ms (256 samples) - float32 [speech] | `onnx` | 16 | 0.464 | 0.382 | 0.914 | 1.652 | `0.0290` | **34x** | ✅ PASS |
| 16ms (256 samples) - pcm_bytes [speech] | `onnx` | 16 | 0.351 | 0.319 | 0.515 | 0.792 | `0.0220` | **46x** | ✅ PASS |
| 32ms (512 samples) - float32 [speech] | `onnx` | 32 | 0.481 | 0.409 | 0.962 | 1.568 | `0.0150` | **66x** | ✅ PASS |
| 32ms (512 samples) - pcm_bytes [speech] | `onnx` | 32 | 0.408 | 0.351 | 0.715 | 1.087 | `0.0127` | **78x** | ✅ PASS |
| 64ms (1024 samples) - float32 [speech] | `onnx` | 64 | 0.608 | 0.517 | 1.044 | 2.011 | `0.0095` | **105x** | ✅ PASS |
| 64ms (1024 samples) - pcm_bytes [speech] | `onnx` | 64 | 0.683 | 0.576 | 1.185 | 2.300 | `0.0107` | **94x** | ✅ PASS |
| 100ms (1600 samples) - float32 [speech] | `onnx` | 100 | 1.007 | 0.881 | 1.821 | 3.252 | `0.0101` | **99x** | ✅ PASS |
| 100ms (1600 samples) - pcm_bytes [speech] | `onnx` | 100 | 1.273 | 1.082 | 2.799 | 7.481 | `0.0127` | **78x** | ✅ PASS |
| 250ms (4000 samples) - float32 [speech] | `onnx` | 250 | 1.851 | 1.648 | 3.149 | 4.668 | `0.0074` | **135x** | ✅ PASS |
| 250ms (4000 samples) - pcm_bytes [speech] | `onnx` | 250 | 1.440 | 1.242 | 2.318 | 6.872 | `0.0058` | **174x** | ✅ PASS |
| 500ms (8000 samples) - float32 [speech] | `onnx` | 500 | 2.855 | 2.672 | 4.198 | 4.934 | `0.0057` | **175x** | ✅ PASS |
| 500ms (8000 samples) - pcm_bytes [speech] | `onnx` | 500 | 2.485 | 2.368 | 3.331 | 4.839 | `0.0050` | **201x** | ✅ PASS |

## 3. Streaming Conversational Session Evaluation
- **Total Session Duration**: `10.0s`
- **Speech Duration**: `6.2s`
- **Silence Duration**: `3.8s`
- **Speech Onset Pulses**: `2`
- **Speech End Transitions**: `2`
- **Streaming Mean Latency**: `0.673ms`
- **Streaming p95 Latency**: `0.986ms`
- **Streaming Max Latency**: `1.524ms`
