"""test_intent_engine.py — Comprehensive Unit & Integration Tests for Part G (Intent Finalization / Smart Listener)."""

import pytest
from intent_engine import intent_engine, IntentEngine, IntentAction, IntentSessionTracker
from refiner_service import refiner_service


def test_explicit_discard_signal_english():
    text = "We should probably go with option A... actually cancel that"
    res = intent_engine.parse_utterance(text)
    assert res.action == IntentAction.DISCARD
    assert res.intended_text == ""
    assert res.discarded_text is not None


def test_explicit_discard_signal_urdu():
    text = "Kal subah meeting rakhte hain... choro isko rehne do"
    res = intent_engine.parse_utterance(text, language="ur_roman")
    assert res.action == IntentAction.DISCARD
    assert res.intended_text == ""


def test_explicit_commit_signal_english():
    text = "So I was thinking about stuff, just type: please review the pull request"
    res = intent_engine.parse_utterance(text)
    assert res.action == IntentAction.COMMIT
    assert res.intended_text == "please review the pull request"
    assert "thinking about stuff" in (res.discarded_text or "")


def test_explicit_commit_signal_urdu():
    text = "Pehle mera khayal tha kuch aur, bas ye likho: hum kal milte hain"
    res = intent_engine.parse_utterance(text, language="ur_roman")
    assert res.action == IntentAction.COMMIT
    assert res.intended_text == "hum kal milte hain"


def test_user_exact_example_flow():
    # Input speech: "I want to go to— there, nowhere, cancel it, just type hi how are you"
    # Expected output: "hi how are you"
    raw = "I want to go to— there, nowhere, cancel it, just type hi how are you"
    res = intent_engine.parse_utterance(raw)
    assert res.action == IntentAction.COMMIT
    assert res.intended_text == "hi how are you"
    assert "nowhere" in (res.discarded_text or "")


@pytest.mark.asyncio
async def test_full_refiner_pipeline_intent_discard():
    # When speech is purely cancelled, the refiner returns empty text and intent_action='discard'
    raw = "I was thinking we could reschedule... cancel that"
    resp = await refiner_service.refine(
        raw_text=raw,
        use_llm=False,
        record_memory=False,
    )
    assert resp.text == ""
    assert resp.intent_action == "discard"
    assert resp.discarded_text is not None


@pytest.mark.asyncio
async def test_full_refiner_pipeline_intent_commit():
    # Full pipeline: extracts intended payload, cleans it, and punctuates it
    raw = "I want to go to— there, nowhere, cancel it, just type hi how are you"
    resp = await refiner_service.refine(
        raw_text=raw,
        use_llm=False,
        record_memory=False,
        punctuation_style="standard",
    )
    assert resp.intent_action == "commit"
    assert "Hi, how are you?" in resp.text
    assert "nowhere" not in resp.text


def test_incomplete_trailing_conjunction():
    text = "We should push the update today because"
    res = intent_engine.parse_utterance(text)
    assert res.action == IntentAction.PENDING
    assert res.is_complete is False


def test_intent_session_tracker_workflow():
    tracker = IntentSessionTracker()

    # Fragment 1: exploratory rambling
    r1 = tracker.process_final("So um I was wondering if we should maybe")
    assert tracker.intent_buffer == ["So um I was wondering if we should maybe"]

    # Fragment 2: cancel signal wipes buffer!
    r2 = tracker.process_final("cancel that, nevermind")
    assert r2.action == IntentAction.DISCARD
    assert len(tracker.intent_buffer) == 0

    # Fragment 3: fresh thought committed
    r3 = tracker.process_final("just type: meeting at 5 PM")
    assert r3.action == IntentAction.COMMIT
    assert tracker.get_commit_payload() == "meeting at 5 PM"
