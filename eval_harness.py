"""eval_harness.py — Automated Evaluation & Quality Benchmarking Harness for VoxFlow v3.

Measures:
1. Intent Classification Accuracy (False Cancel Rate & False Commit Rate)
2. Content/Meaning Preservation Score (Guardrail verification)
3. Forbidden Word Elimination (Fillers, stutters, cancelled fragments)
4. Overall End-to-End Success Rate across categories
"""

from __future__ import annotations

import asyncio
import json
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

from intent_engine import intent_engine, IntentAction
from refiner_service import refiner_service

DATASET_PATH = Path(__file__).resolve().parent / "voxflow" / "data" / "eval_dataset.json"


async def run_evaluation():
    if not DATASET_PATH.exists():
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    print("=" * 70)
    print(f"VOXFLOW v3 BENCHMARK & EVALUATION HARNESS ({len(cases)} test cases)")
    print("=" * 70)

    total_cases = len(cases)
    passed_cases = 0
    false_cancels = 0
    false_commits = 0
    meaning_scores: list[float] = []

    for c in cases:
        case_id = c["id"]
        category = c["category"]
        raw_input = c["input"]
        expected_action = c["expected_action"]
        expected_contains = c.get("expected_text_contains", "")
        forbidden = c.get("forbidden_words", [])
        silence_ms = c.get("silence_duration_ms", 0.0)

        # 1. Intent Decision
        intent_res = intent_engine.parse_utterance(raw_input, silence_duration_ms=silence_ms)
        actual_action = intent_res.action.value

        # Check intent errors
        if expected_action != "discard" and actual_action == "discard":
            false_cancels += 1
            action_match = False
        elif expected_action != "commit" and actual_action == "commit" and expected_action != "pass":
            false_commits += 1
            action_match = False
        else:
            action_match = (actual_action == expected_action) or (expected_action in ("pass", "commit") and actual_action in ("pass", "commit"))

        # 2. Pipeline Refinement (unless pure discard)
        if actual_action == "discard":
            final_text = ""
            meaning_score = 1.0
        else:
            refine_res = await refiner_service.refine(
                raw_text=raw_input,
                use_llm=True,
                record_memory=False,
                silence_duration_ms=silence_ms,
            )
            final_text = refine_res.text
            meaning_score = refine_res.intent_score

        meaning_scores.append(meaning_score)

        # 3. Content assertions
        content_match = True
        if expected_contains and expected_contains.lower() not in final_text.lower():
            content_match = False

        forbidden_clean = True
        for fw in forbidden:
            if fw.lower() in final_text.lower():
                forbidden_clean = False
                break

        case_passed = action_match and content_match and forbidden_clean
        if case_passed:
            passed_cases += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"[{status}] {case_id:<28} | Act: {actual_action:<7} (exp {expected_action}) | Score: {meaning_score:.2f} | Final: {final_text[:35]}...")

    pass_rate = (passed_cases / total_cases) * 100.0
    fc_rate = (false_cancels / total_cases) * 100.0
    fcm_rate = (false_commits / total_cases) * 100.0
    avg_meaning = sum(meaning_scores) / max(1, len(meaning_scores))

    print("\n" + "=" * 70)
    print("EVALUATION METRICS SUMMARY:")
    print(f"  Total Test Cases:            {total_cases}")
    print(f"  Passed Cases:                {passed_cases}/{total_cases} ({pass_rate:.1f}%)")
    print(f"  False-Cancel Rate:           {fc_rate:.1f}% (0.0% is optimal)")
    print(f"  False-Commit Rate:           {fcm_rate:.1f}% (0.0% is optimal)")
    print(f"  Avg Meaning Preservation:    {avg_meaning:.3f} (1.000 max)")
    print("=" * 70)

    # Return exit code 0 if pass rate >= 90%
    return pass_rate >= 90.0


if __name__ == "__main__":
    success = asyncio.run(run_evaluation())
    sys.exit(0 if success else 1)
