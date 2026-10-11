"""Tests for transcript_refiner.py.

All Gemini SDK calls are mocked so the tests run without a network or API key.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
import sys
from unittest.mock import MagicMock
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import transcript_refiner

MODULE = transcript_refiner
llm_refine = transcript_refiner.llm_refine
TRANSCRIPT_CLEANUP_PROMPT = transcript_refiner.TRANSCRIPT_CLEANUP_PROMPT


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_mock_model(refined_text: str):
    """Return a MagicMock that mimics a Gemini GenerativeModel."""
    mock_response = MagicMock()
    mock_response.text = refined_text

    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response
    return mock_model


# ── Prompt template ───────────────────────────────────────────────────────────

def test_prompt_contains_placeholder():
    """The system prompt must contain the {raw_transcript} placeholder."""
    assert "{raw_transcript}" in TRANSCRIPT_CLEANUP_PROMPT


def test_prompt_substitution():
    """Substituting a sample transcript produces a prompt containing that text."""
    sample = "um I wanna deploy the new feature"
    rendered = TRANSCRIPT_CLEANUP_PROMPT.format(raw_transcript=sample)
    assert sample in rendered


def test_prompt_contains_key_rules():
    """Key rule headings from the spec must be present in the prompt."""
    for keyword in (
        "Filler",
        "Self-Corrections",
        "De-duplicate",
        "Preserve Voice",
        "Zero Hallucination",
    ):
        assert keyword in TRANSCRIPT_CLEANUP_PROMPT, (
            f"Expected keyword '{keyword}' not found in TRANSCRIPT_CLEANUP_PROMPT"
        )


# ── Fallback: no API key ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_returns_input_when_no_api_key(monkeypatch):
    """llm_refine must return the input text unchanged when GEMINI_API_KEY is absent."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    # Reset the cached client so the env change takes effect.
    MODULE._client = None

    text = "We need to ship the hotfix today."
    result = await llm_refine(text)
    assert result == text


@pytest.mark.asyncio
async def test_returns_input_for_empty_string(monkeypatch):
    """llm_refine must return the empty string unchanged (no API call)."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    MODULE._client = None

    result = await llm_refine("")
    assert result == ""


# ── Success path (mocked Gemini) ───────────────────────────────────────────────

@pytest.mark.asyncio
async def test_returns_llm_refined_text(monkeypatch):
    """When Gemini returns refined text, llm_refine must return that text."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-abc")
    refined = "We need to deploy the hotfix today."
    MODULE._client = _make_mock_model(refined)

    raw = "um we gotta like deploy the, you know, hotfix today."
    result = await llm_refine(raw)
    assert result == refined


@pytest.mark.asyncio
async def test_llm_call_receives_full_prompt(monkeypatch):
    """generate_content must be called with a prompt containing the input text."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-abc")
    mock_model = _make_mock_model("Refined output.")
    MODULE._client = mock_model

    raw = "so basically the cache is broken."
    await llm_refine(raw)

    call_args = mock_model.generate_content.call_args
    assert call_args is not None
    prompt_used: str = call_args[0][0]
    assert raw in prompt_used
    assert "{raw_transcript}" not in prompt_used, (
        "Placeholder was not substituted in the prompt sent to the LLM"
    )


# ── Fallback: API errors ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_falls_back_on_api_exception(monkeypatch):
    """If generate_content raises, llm_refine must return the original text."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-abc")

    mock_model = MagicMock()
    mock_model.generate_content.side_effect = RuntimeError("quota exceeded")
    MODULE._client = mock_model

    text = "Deploy the hotfix now."
    result = await llm_refine(text)
    assert result == text


@pytest.mark.asyncio
async def test_falls_back_on_empty_response(monkeypatch):
    """If Gemini returns an empty string, llm_refine must return the original text."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-abc")
    MODULE._client = _make_mock_model("")  # empty response

    text = "Deploy the hotfix now."
    result = await llm_refine(text)
    assert result == text
