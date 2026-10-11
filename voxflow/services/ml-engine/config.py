"""config.py — Centralized configuration management for VoxFlow ML Engine.

Uses standard Pydantic and python-dotenv for environment-based configuration
without requiring additional third-party dependencies.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env file from the service directory if present
_ENV_PATH = Path(__file__).resolve().parent / ".env"
if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)
else:
    load_dotenv()


class Settings(BaseModel):
    # Server configuration
    host: str = Field(default_factory=lambda: os.getenv("HOST", "127.0.0.1"))
    port: int = Field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    debug: bool = Field(default_factory=lambda: os.getenv("DEBUG", "false").lower() in ("true", "1", "yes"))

    # Security & Limits
    allowed_origins: list[str] = Field(
        default_factory=lambda: [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",")]
    )
    max_transcript_length: int = Field(
        default_factory=lambda: int(os.getenv("MAX_TRANSCRIPT_LENGTH", "8192"))
    )
    request_timeout_seconds: float = Field(
        default_factory=lambda: float(os.getenv("REQUEST_TIMEOUT_SECONDS", "15.0"))
    )

    # Local Engine & On-Device Models
    local_model_path: Path | None = Field(default=None)
    asr_model_size: str = Field(default_factory=lambda: os.getenv("ASR_MODEL_SIZE", "base"))
    vad_threshold: float = Field(default_factory=lambda: float(os.getenv("VAD_THRESHOLD", "0.5")))
    vad_silence_duration_ms: float = Field(default_factory=lambda: float(os.getenv("VAD_SILENCE_MS", "1200.0")))
    offline_mode_only: bool = Field(default=True)

    # Deprecated Cloud Settings (Maintained as blank for backward-compatibility)
    gemini_api_key: str = Field(default="")
    gemini_model_name: str = Field(default="disabled-local-only")

    # Adaptive Memory
    storage_path: Path = Field(
        default_factory=lambda: Path(os.getenv("STORAGE_PATH", str(Path(__file__).resolve().parent / "adaptive_memory.json")))
    )
    filler_threshold: int = Field(
        default_factory=lambda: int(os.getenv("FILLER_THRESHOLD", "3"))
    )

    # Background Job Engine
    job_workers: int = Field(
        default_factory=lambda: int(os.getenv("JOB_WORKERS", "3"))
    )
    job_max_retries: int = Field(
        default_factory=lambda: int(os.getenv("JOB_MAX_RETRIES", "3"))
    )
    job_backoff_base: float = Field(
        default_factory=lambda: float(os.getenv("JOB_BACKOFF_BASE", "1.5"))
    )
    job_ttl_seconds: int = Field(
        default_factory=lambda: int(os.getenv("JOB_TTL_SECONDS", "3600"))
    )

    # In-memory Caching
    cache_max_size: int = Field(
        default_factory=lambda: int(os.getenv("CACHE_MAX_SIZE", "1024"))
    )


# Global singleton instance
settings = Settings()
