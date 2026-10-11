"""main.py — VoxFlow Enterprise ML Engine & Voice Processing Gateway.

Features:
- FastAPI gateway with CORS, Request Timing, and Global Error Handling
- WebSocket endpoint for low-latency streaming audio & live transcript refinement
- Asynchronous Background Job Engine (queue, worker pool, backoff retries, DLQ)
- LRU Caching Layer for sub-millisecond transcript processing
- Adaptive Memory Engine tracking personal vocal fillers
- 100% on-device intent-preserving local refinement (no cloud LLM)
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Ensure current directory is in sys.path
_CURRENT_DIR = str(Path(__file__).resolve().parent)
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import ValidationError

from config import settings
from cache import transcript_cache
from job_manager import job_manager, JobStatus
from refiner_service import refiner_service, RefinerService
from cleanup_engine import cleanup_engine
from schemas import (
    TranscriptRequest,
    TranscriptResponse,
    JobSubmissionRequest,
    JobResponse,
    HealthResponse,
    MemorySummaryResponse,
    CorrectionFeedbackRequest,
    CorrectionFeedbackResponse,
    PersonalizationMetricsResponse,
    VocabularyRequest,
)
from personalization_service import personalization_service

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
log = logging.getLogger("voxflow")

# Backwards compatibility constants & symbols
UI_PATH = Path(__file__).resolve().with_name("web_ui.html")
WIDGET_PATH = Path(__file__).resolve().with_name("widget.html")
MEMORY = refiner_service.memory
clean_transcript = RefinerService.clean_transcript
best_word = RefinerService.best_word

APP_VERSION = "3.0.0"
_SERVER_START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager: initialize background job queue & workers on startup."""
    log.info("Starting VoxFlow ML Engine service (port %d)...", settings.port)
    job_manager.set_processor(refiner_service.process_job)
    await job_manager.start()
    yield
    log.info("Shutting down VoxFlow ML Engine...")
    await job_manager.shutdown()


app = FastAPI(
    title="VoxFlow ML Engine",
    description="Enterprise voice-first transcript refinement and adaptive memory service.",
    version=APP_VERSION,
    lifespan=lifespan,
)

# ─── Middleware ───────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_and_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Process-Time"] = f"{duration_ms:.2f}ms"
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception):
    log.error("Unhandled server exception: %s", exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error occurred", "type": type(exc).__name__},
    )


# ─── Static Web UI Routes ─────────────────────────────────────────────────────

@app.get("/")
def ui():
    """Interactive VoxFlow Web Studio."""
    return FileResponse(UI_PATH)


@app.get("/widget")
def widget():
    """Standalone Floating Pill Widget with wake-word detection."""
    return FileResponse(WIDGET_PATH)


@app.get("/widget.html")
def widget_html():
    """Alias for /widget."""
    return FileResponse(WIDGET_PATH)


# ─── Core Health & Memory Endpoints ──────────────────────────────────────────

@app.get("/health")
def health(details: bool = Query(default=False)):
    """Health check endpoint. Preserves backward compatibility and returns diagnostics when requested."""
    if not details:
        return {"status": "ok"}
    return health_detailed()


@app.get("/api/v1/health", response_model=HealthResponse)
def health_detailed():
    """Detailed operational health and diagnostic metrics."""
    uptime = time.time() - _SERVER_START_TIME
    return HealthResponse(
        status="ok",
        version=APP_VERSION,
        uptime_seconds=round(uptime, 2),
        job_stats=job_manager.stats(),
        memory_stats=MEMORY.summary(),
        local_pipeline_active=True,
        gemini_configured=False,
    )


@app.get("/memory", response_model=MemorySummaryResponse)
def memory_summary():
    """Adaptive memory statistics showing learned filler words and mined domain bigrams."""
    return MemorySummaryResponse(**MEMORY.summary())


# ─── REST Refinement & Background Job Endpoints ──────────────────────────────

@app.post("/api/v1/refine/sync", response_model=TranscriptResponse)
async def refine_synchronous(req: TranscriptRequest):
    """Synchronous transcript refinement (cached, multi-language, intent-preserving)."""
    return await refiner_service.refine(
        raw_text=req.text,
        use_llm=req.use_llm,
        record_memory=True,
        language=req.language,
        punctuation_style=req.punctuation_style,  # type: ignore[arg-type]
        enable_emoji=req.enable_emoji,
        emoji_theme=req.emoji_theme,
        emoji_density=req.emoji_density,
        output_script=req.output_script,
        silence_duration_ms=req.silence_duration_ms,
    )


@app.post("/api/v1/jobs/refine", response_model=JobResponse, status_code=202)
async def submit_background_job(req: JobSubmissionRequest):
    """Submit a transcript for asynchronous background refinement.

    Returns immediately with a 202 Accepted status and job tracking ID.
    Supports idempotency keys to safely prevent duplicate execution.
    """
    job = await job_manager.submit_job(
        text=req.text,
        idempotency_key=req.idempotency_key,
        use_llm=req.use_llm,
    )
    return job.to_response()


# NOTE: static paths must be registered before "/api/v1/jobs/{job_id}",
# otherwise FastAPI matches "dead-letter" as a job_id and returns 404.
@app.get("/api/v1/jobs/dead-letter", response_model=list[JobResponse])
def get_dead_letter_queue(limit: int = Query(default=50, ge=1, le=200)):
    """Inspect jobs that failed all retry attempts and moved to Dead-Letter Queue."""
    dlq = job_manager.get_dead_letter_queue(limit=limit)
    return [j.to_response() for j in dlq]


@app.get("/api/v1/jobs", response_model=list[JobResponse])
def list_jobs(
    status: JobStatus | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List recent background jobs with optional status filter."""
    jobs = job_manager.list_jobs(status=status, limit=limit, offset=offset)
    return [j.to_response() for j in jobs]


@app.get("/api/v1/jobs/{job_id}", response_model=JobResponse)
def get_job_status(job_id: str):
    """Query status and refined results for a background job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.to_response()


@app.get("/api/v1/cache/stats")
def get_cache_stats():
    """Retrieve LRU transcript cache hit/miss statistics."""
    return transcript_cache.stats()


# ─── Part H: Local Personalization & Adaptation Endpoints ────────────────────

@app.post("/api/v1/personalization/feedback", response_model=CorrectionFeedbackResponse)
def record_correction_feedback(req: CorrectionFeedbackRequest):
    """Log user post-injection manual edits locally for safe adaptive learning."""
    feedback = personalization_service.record_correction(
        raw=req.raw,
        pipeline_output=req.pipeline_output,
        user_edited=req.user_edited,
        app_context=req.app_context,
    )
    return CorrectionFeedbackResponse(
        logged=True,
        character_distance=feedback.character_distance,
        edit_rate=feedback.edit_rate,
        learned_terms_added=sorted(personalization_service.vocabulary),
    )


@app.get("/api/v1/personalization/metrics", response_model=PersonalizationMetricsResponse)
def get_personalization_metrics():
    """Track local edit rate trends and adapter retraining readiness."""
    metrics = personalization_service.get_metrics_summary()
    return PersonalizationMetricsResponse(**metrics)


@app.post("/api/v1/personalization/vocabulary")
def add_personal_vocabulary(req: VocabularyRequest):
    """Add a domain term, jargon, or proper noun to local ASR decoder biasing list."""
    personalization_service.add_vocabulary_term(req.term)
    return {"status": "added", "term": req.term, "total_terms": len(personalization_service.vocabulary)}


@app.get("/api/v1/personalization/vocabulary")
def get_personal_vocabulary():
    """List all user-specific technical terms and proper nouns."""
    return {"vocabulary": sorted(personalization_service.vocabulary)}


# ─── Real-Time WebSocket Gateway ──────────────────────────────────────────────

def _validation_message(err: ValidationError) -> str:
    """Condense a Pydantic ValidationError into a short client-facing message."""
    first = err.errors()[0] if err.errors() else {}
    field = ".".join(str(p) for p in first.get("loc", ())) or "payload"
    return f"invalid '{field}': {first.get('msg', 'validation failed')}"


async def _send_error(websocket: WebSocket, message: str, **extra: Any) -> None:
    await websocket.send_text(json.dumps({"type": "error", "message": message, **extra}))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_text()
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                await _send_error(websocket, "invalid json")
                continue

            if not isinstance(payload, dict):
                await _send_error(websocket, "payload must be a JSON object")
                continue

            msg_type = payload.get("type")

            if msg_type == "audio":
                # Audio acknowledgement with byte count
                data_str = payload.get("data", "")
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "ack",
                            "received": True,
                            "bytes": len(data_str),
                        }
                    )
                )

            elif msg_type == "ping":
                # Heartbeat
                await websocket.send_text(json.dumps({"type": "pong"}))

            elif msg_type == "transcript_interim":
                # Tier 1 fast lightweight pass for live interim preview (<0.2ms)
                raw_text = payload.get("text") or ""
                if not isinstance(raw_text, str):
                    await _send_error(websocket, "invalid 'text': must be a string")
                    continue
                interim_clean = cleanup_engine.clean_interim(raw_text)
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "transcript_interim",
                            "raw": raw_text,
                            "text": interim_clean,
                        }
                    )
                )

            elif msg_type == "transcript":
                # Full multi-stage speech processing for finalized utterance
                raw_text = payload.get("text") or ""
                if isinstance(raw_text, str) and not raw_text.strip():
                    continue  # silently ignore empty utterances (e.g. VAD noise)

                try:
                    req = TranscriptRequest.model_validate(payload)
                except ValidationError as verr:
                    await _send_error(websocket, _validation_message(verr), raw=str(raw_text)[:200])
                    continue

                try:
                    response = await refiner_service.refine(
                        raw_text=req.text,
                        use_llm=req.use_llm,
                        record_memory=True,
                        language=req.language,
                        punctuation_style=req.punctuation_style,  # type: ignore[arg-type]
                        enable_emoji=req.enable_emoji,
                        emoji_theme=req.emoji_theme,
                        emoji_density=req.emoji_density,
                        output_script=req.output_script,
                        silence_duration_ms=req.silence_duration_ms,
                    )

                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "transcript",
                                "raw": response.raw,
                                "text_baseline": response.text_baseline,
                                "text": response.text,
                                "best_word": response.best_word,
                                "memory": response.memory.model_dump(),
                                "cached": response.cached,
                                "duration_ms": response.duration_ms,
                                "detected_language": response.detected_language,
                                "language_confidence": response.language_confidence,
                                "is_code_switched": response.is_code_switched,
                                "emoji_inserted": response.emoji_inserted,
                                "emotion_detected": response.emotion_detected,
                                "valence": response.valence,
                                "intent_score": response.intent_score,
                                "drift_detected": response.drift_detected,
                                "intent_action": response.intent_action,
                                "discarded_text": response.discarded_text,
                                "is_complete_intent": response.is_complete_intent,
                            }
                        )
                    )
                except Exception as err:
                    log.error("WebSocket transcript refinement error: %s", err, exc_info=True)
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "error",
                                "message": f"Refinement error: {str(err)}",
                                "raw": raw_text,
                            }
                        )
                    )

            elif msg_type == "job_submit":
                # Asynchronous background job submission via WebSocket
                try:
                    job_req = JobSubmissionRequest.model_validate(payload)
                except ValidationError as verr:
                    await _send_error(websocket, _validation_message(verr))
                    continue
                job = await job_manager.submit_job(
                    text=job_req.text,
                    idempotency_key=job_req.idempotency_key,
                    use_llm=job_req.use_llm,
                )
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "job_enqueued",
                            "job": job.to_response().model_dump(),
                        }
                    )
                )

            else:
                await websocket.send_text(json.dumps({"type": "echo", "payload": payload}))

    except WebSocketDisconnect:
        return


def start(host: str = "127.0.0.1", port: int = 8000):
    """Start the VoxFlow ML Engine uvicorn server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start(host=settings.host, port=settings.port)
