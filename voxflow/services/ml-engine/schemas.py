"""schemas.py — Typed Pydantic data transfer models for VoxFlow.

Provides strict validation for REST endpoints, WebSockets, and background jobs.
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"


class TranscriptRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=16384, description="Raw spoken transcript")
    idempotency_key: str | None = Field(default=None, max_length=128, description="Optional unique key to ensure idempotent processing")
    use_llm: bool = Field(default=True, description="Whether to apply the on-device deep refinement stage (self-corrections, structuring)")
    language: str | None = Field(default=None, description="Optional forced language code, or None for auto-detection")
    punctuation_style: str = Field(default="standard", description="Punctuation style: 'casual', 'standard', or 'formal'")
    enable_emoji: bool = Field(default=False, description="Toggle contextual emoji insertion (default False)")
    emoji_theme: str | None = Field(default="standard", description="Emoji theme: 'standard', 'work', or 'playful'")
    emoji_density: str | None = Field(default="balanced", description="Emoji density: 'subtle', 'balanced', or 'expressive'")
    output_script: str | None = Field(default=None, description="Script preference: 'latin' or 'arabic'")
    silence_duration_ms: float = Field(default=0.0, ge=0.0, description="VAD-measured silence duration in milliseconds")


class MemorySummaryResponse(BaseModel):
    transcript_count: int
    learned_fillers: dict[str, int]
    common_phrases: list[str]


class TranscriptResponse(BaseModel):
    type: str = "transcript"
    raw: str
    text_baseline: str
    text: str
    best_word: str
    memory: MemorySummaryResponse
    cached: bool = False
    duration_ms: float | None = None
    # Enhanced ML & Language Metadata
    detected_language: str = "en"
    language_confidence: float = 1.0
    is_code_switched: bool = False
    emoji_inserted: str | None = None
    emotion_detected: str | None = None
    valence: str = "neutral"
    intent_score: float = 1.0
    drift_detected: bool = False
    # Part G: Intent Finalization & Smart Listener
    intent_action: str = "pass"
    discarded_text: str | None = None
    is_complete_intent: bool = True


class JobSubmissionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=16384, description="Transcript text to refine asynchronously")
    idempotency_key: str | None = Field(default=None, max_length=128, description="Optional idempotency key to prevent duplicate processing")
    use_llm: bool = Field(default=True, description="Enable the on-device deep refinement stage")
    language: str | None = Field(default=None, description="Optional forced language code")
    punctuation_style: str = Field(default="standard", description="Punctuation style")
    enable_emoji: bool = Field(default=False, description="Toggle contextual emoji insertion")


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    idempotency_key: str | None = None
    created_at: float
    started_at: float | None = None
    completed_at: float | None = None
    retries_attempted: int = 0
    raw_text: str
    result: TranscriptResponse | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str = "3.0.0"
    uptime_seconds: float
    job_stats: dict[str, Any]
    memory_stats: dict[str, Any]
    local_pipeline_active: bool = True
    gemini_configured: bool = False  # deprecated: always False since v3 (kept for API compatibility)


class CorrectionFeedbackRequest(BaseModel):
    raw: str = Field(..., description="Original raw transcript input")
    pipeline_output: str = Field(..., description="Refined text output by pipeline")
    user_edited: str = Field(..., description="Final text manually edited by user")
    app_context: str | None = Field(default=None, description="Active application name")


class CorrectionFeedbackResponse(BaseModel):
    logged: bool
    character_distance: int
    edit_rate: float
    learned_terms_added: list[str] = Field(default_factory=list)


class PersonalizationMetricsResponse(BaseModel):
    total_corrections: int
    average_edit_rate: float
    recent_edit_rate: float
    vocabulary_count: int
    ready_for_adapter_training: bool


class VocabularyRequest(BaseModel):
    term: str = Field(..., min_length=2, max_length=64, description="Technical term, acronym, or proper noun")
