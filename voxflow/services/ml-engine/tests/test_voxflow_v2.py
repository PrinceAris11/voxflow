"""test_voxflow_v2.py — Test suite for Language ID, Cleanup, Punctuation, and Emojis."""

import pytest
from cleanup_engine import cleanup_engine
from emoji_service import emoji_service
from language_detector import language_detector
from punctuation_service import punctuation_service
from refiner_service import refiner_service


# ─── Part B: Language Identification & Code-Switching ─────────────────────────

def test_language_detection_english():
    res = language_detector.detect("Please review the pull request and deploy the build.")
    assert res.language == "en"
    assert res.confidence >= 0.6
    assert not res.is_code_switched


def test_language_detection_urdu_script():
    res = language_detector.detect("یہ ایک نیا وائس پروجیکٹ ہے")
    assert res.language == "ur"
    assert res.output_script == "arabic"
    assert not res.is_code_switched


def test_language_detection_roman_urdu():
    res = language_detector.detect("yaar mujhe ye kaam kal tak karke dena hai")
    assert res.language in ("ur_roman", "en_ur_mixed")
    assert res.output_script == "latin"


def test_language_detection_code_switching():
    res = language_detector.detect("yaar mujhe ye file today submit karni hai")
    assert res.is_code_switched is True
    assert res.language in ("en_ur_mixed", "ur_roman")


def test_language_detection_fallback_on_short_ambiguous():
    detector = language_detector.__class__(default_language="en")
    res1 = detector.detect("Hello world")
    assert res1.language == "en"

    # Ambiguous punctuation/number
    res2 = detector.detect("123 456")
    assert res2.language == "en"


# ─── Part C: ML Cleanup & Intent Guardrails ────────────────────────────────────

def test_comparison_like_is_preserved():
    res = cleanup_engine.clean_finalized("I want a car like a Tesla")
    assert "like a Tesla" in res.cleaned_text


def test_hesitation_like_is_removed():
    res = cleanup_engine.clean_finalized("So um like we should fix this bug")
    assert "like" not in res.cleaned_text.lower()
    assert "we should fix this bug" in res.cleaned_text.lower()


def test_self_correction_resolution():
    res = cleanup_engine.clean_finalized("We should meet at 3 PM... no wait, make it 4 PM")
    assert "4 PM" in res.cleaned_text
    assert "3 PM" not in res.cleaned_text


def test_stutter_repetition_collapsed():
    res = cleanup_engine.clean_finalized("I I want to to submit the the report")
    assert res.cleaned_text.lower().startswith("i want to submit the report")


def test_intentional_emphasis_preserved():
    res = cleanup_engine.clean_finalized("This is very very important")
    assert "very very" in res.cleaned_text.lower()


def test_roman_urdu_fillers_removed():
    res = cleanup_engine.clean_finalized(
        "yaar matlab ke mujhe ye file submit karni hai",
        language="ur_roman",
    )
    assert "matlab ke" not in res.cleaned_text.lower()
    assert "mujhe ye file submit karni hai" in res.cleaned_text.lower()


def test_intent_guardrail_preserves_facts_and_numbers():
    res = cleanup_engine.clean_finalized("Database latency jumped to 450 ms in production")
    assert "450" in res.cleaned_text
    assert res.intent_preservation_score >= 0.70
    assert res.drift_detected is False


# ─── Part D: Punctuation Restoration ──────────────────────────────────────────

def test_punctuation_question_intonation():
    p1 = punctuation_service.punctuate("what time is the deployment meeting")
    assert p1.endswith("?")

    p2 = punctuation_service.punctuate("can you review the PR before noon")
    assert p2.endswith("?")


def test_punctuation_tag_question_casual():
    p = punctuation_service.punctuate("youre coming right")
    assert "right?" in p
    assert "," in p


def test_punctuation_tech_acronyms_capitalized():
    p = punctuation_service.punctuate("we need to optimize the api endpoint in redis and docker")
    assert "API" in p
    assert "Redis" in p
    assert "Docker" in p


def test_punctuation_transitional_commas():
    p = punctuation_service.punctuate("however we should verify the staging build")
    assert p.startswith("However,")


# ─── Part E: Emotion-Based Emoji Insertion ─────────────────────────────────────

def test_emoji_disabled_by_default():
    text = "I am so excited about this huge launch!"
    res = emoji_service.process(text, enabled=False)
    assert res.final_text == text
    assert res.emoji_inserted is None


def test_emoji_enabled_appends_at_end():
    text = "I am so thrilled about our new product!"
    res = emoji_service.process(text, enabled=True)
    assert res.emoji_inserted is not None
    assert res.final_text.endswith(res.emoji_inserted)


def test_emoji_skips_technical_instructions():
    text = "We should update the Redis cache and restart Docker."
    res = emoji_service.process(text, enabled=True)
    assert res.emoji_inserted is None
    assert res.final_text == text


def test_emoji_urdu_excitement():
    text = "Zabardast kaam kiya bhai!"
    res = emoji_service.process(text, enabled=True, language="ur_roman")
    assert res.emoji_inserted is not None


def test_emoji_negation_inversion():
    text = "I am not happy with this result at all."
    res = emoji_service.process(text, enabled=True)
    assert res.detected_emotion != "happy"
    assert res.negation_detected is True
    assert res.valence != "positive"


def test_emoji_somber_suppression():
    text = "Please send my condolences, his grandfather passed away."
    res = emoji_service.process(text, enabled=True)
    assert res.emoji_inserted is None
    assert res.suppression_reason == "somber_context_suppressed"


def test_emoji_native_urdu_script():
    text = "ہم جیت گئے، بہت مبارک ہو!"
    res = emoji_service.process(text, enabled=True, language="ur")
    assert res.emoji_inserted is not None
    assert res.detected_emotion == "celebratory"


def test_emoji_themes_work_vs_playful():
    text = "We won the championship game!"
    res_work = emoji_service.process(text, enabled=True, theme="work")
    res_playful = emoji_service.process(text, enabled=True, theme="playful")
    assert res_work.emoji_inserted in ["🎉", "👏", "🏆"]
    assert res_playful.emoji_inserted is not None


def test_emoji_deterministic_selection():
    text = "Let's go team, awesome work!"
    res1 = emoji_service.process(text, enabled=True)
    res2 = emoji_service.process(text, enabled=True)
    assert res1.emoji_inserted == res2.emoji_inserted
    assert res1.final_text == res2.final_text


# ─── Integration: Full Refiner Pipeline ────────────────────────────────────────

@pytest.mark.asyncio
async def test_full_refiner_pipeline_end_to_end():
    raw = "So um basically I want a car like a Tesla... no wait, make it a Rivian, you know?"
    resp = await refiner_service.refine(
        raw_text=raw,
        use_llm=False,
        record_memory=False,
        punctuation_style="standard",
        enable_emoji=False,
    )
    assert "like a Rivian" in resp.text
    assert "Tesla" not in resp.text
    assert resp.detected_language in ("en", "ur_roman", "en_ur_mixed")
    assert resp.intent_score >= 0.50
    assert resp.cached is False
