"""job_manager.py — Production-grade Asynchronous Background Job Engine.

Implements an asynchronous task queue with:
- Job status lifecycle: PENDING -> RUNNING -> COMPLETED / RETRYING -> FAILED
- Worker pool coroutines processing jobs asynchronously
- Idempotency key tracking to avoid duplicate execution
- Exponential backoff retry logic for transient LLM/network failures
- Dead-Letter Queue (DLQ) for jobs that exhaust all retry attempts
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Callable, Coroutine
from schemas import JobStatus, JobResponse, TranscriptResponse

log = logging.getLogger(__name__)


class Job:
    """Internal job state tracking unit."""

    def __init__(
        self,
        text: str,
        idempotency_key: str | None = None,
        use_llm: bool = True,
        max_retries: int = 3,
        backoff_base: float = 1.5,
    ):
        self.job_id: str = str(uuid.uuid4())
        self.idempotency_key: str | None = idempotency_key
        self.raw_text: str = text
        self.use_llm: bool = use_llm
        self.status: JobStatus = JobStatus.PENDING
        self.created_at: float = time.time()
        self.started_at: float | None = None
        self.completed_at: float | None = None
        self.retries_attempted: int = 0
        self.max_retries: int = max_retries
        self.backoff_base: float = backoff_base
        self.result: Any | None = None
        self.error: str | None = None
        self.error_history: list[str] = []

    def to_response(self) -> JobResponse:
        res_payload = None
        if self.result:
            if isinstance(self.result, dict):
                res_payload = TranscriptResponse(**self.result)
            elif isinstance(self.result, TranscriptResponse):
                res_payload = self.result

        return JobResponse(
            job_id=self.job_id,
            status=self.status,
            idempotency_key=self.idempotency_key,
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            retries_attempted=self.retries_attempted,
            raw_text=self.raw_text,
            result=res_payload,
            error=self.error,
        )


class JobManager:
    """Production asynchronous job queue manager."""

    def __init__(
        self,
        worker_count: int = 3,
        max_retries: int = 3,
        backoff_base: float = 1.5,
    ):
        self.worker_count = worker_count
        self.max_retries = max_retries
        self.backoff_base = backoff_base

        self._jobs: dict[str, Job] = {}
        self._idempotency_map: dict[str, str] = {}  # idempotency_key -> job_id
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._dead_letter_queue: list[Job] = []

        self._workers: list[asyncio.Task[None]] = []
        self._processor: Callable[[Job], Coroutine[Any, Any, Any]] | None = None
        self._running: bool = False
        self._lock = asyncio.Lock()

        # Operational metrics
        self._total_submitted = 0
        self._total_completed = 0
        self._total_retried = 0
        self._total_failed = 0

    def set_processor(
        self, processor_fn: Callable[[Job], Coroutine[Any, Any, Any]]
    ) -> None:
        """Register the async function that processes a single job."""
        self._processor = processor_fn

    async def start(self) -> None:
        """Start the background worker tasks."""
        if self._running:
            return
        self._running = True
        log.info(
            "Starting JobManager background engine with %d workers",
            self.worker_count,
        )
        for i in range(self.worker_count):
            task = asyncio.create_task(self._worker_loop(i))
            self._workers.append(task)

    async def shutdown(self) -> None:
        """Gracefully stop all workers."""
        if not self._running:
            return
        self._running = False
        log.info("Shutting down JobManager background workers...")
        for task in self._workers:
            task.cancel()
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def submit_job(
        self,
        text: str,
        idempotency_key: str | None = None,
        use_llm: bool = True,
    ) -> Job:
        """Submit a new job.

        If an idempotency_key is provided and a job is already PENDING,
        RUNNING, or COMPLETED, the existing job is returned idempotently.
        """
        async with self._lock:
            if idempotency_key:
                existing_id = self._idempotency_map.get(idempotency_key)
                if existing_id and existing_id in self._jobs:
                    existing_job = self._jobs[existing_id]
                    if existing_job.status in (
                        JobStatus.PENDING,
                        JobStatus.RUNNING,
                        JobStatus.COMPLETED,
                        JobStatus.RETRYING,
                    ):
                        log.info(
                            "Idempotency hit for key %s -> Job %s (status: %s)",
                            idempotency_key,
                            existing_job.job_id,
                            existing_job.status,
                        )
                        return existing_job

            job = Job(
                text=text,
                idempotency_key=idempotency_key,
                use_llm=use_llm,
                max_retries=self.max_retries,
                backoff_base=self.backoff_base,
            )
            self._jobs[job.job_id] = job
            if idempotency_key:
                self._idempotency_map[idempotency_key] = job.job_id

            self._total_submitted += 1
            await self._queue.put(job.job_id)
            log.info("Enqueued job %s (queue depth: %d)", job.job_id, self._queue.qsize())
            return job

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        status: JobStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Job]:
        all_jobs = list(self._jobs.values())
        if status:
            all_jobs = [j for j in all_jobs if j.status == status]
        all_jobs.sort(key=lambda j: j.created_at, reverse=True)
        return all_jobs[offset : offset + limit]

    def get_dead_letter_queue(self, limit: int = 50) -> list[Job]:
        return self._dead_letter_queue[-limit:]

    async def _worker_loop(self, worker_id: int) -> None:
        """Asynchronous worker loop pulling jobs from the internal queue."""
        log.debug("Worker-%d started", worker_id)
        while self._running:
            try:
                job_id = await self._queue.get()
            except asyncio.CancelledError:
                break

            job = self._jobs.get(job_id)
            if not job:
                self._queue.task_done()
                continue

            await self._execute_job(job, worker_id)
            self._queue.task_done()

    async def _execute_job(self, job: Job, worker_id: int) -> None:
        """Execute the job with exponential backoff retries."""
        job.status = JobStatus.RUNNING
        if job.started_at is None:
            job.started_at = time.time()

        if not self._processor:
            job.status = JobStatus.FAILED
            job.error = "No processor registered with JobManager"
            job.completed_at = time.time()
            return

        try:
            log.info("Worker-%d processing job %s (attempt %d)", worker_id, job.job_id, job.retries_attempted + 1)
            result = await self._processor(job)
            job.result = result
            job.status = JobStatus.COMPLETED
            job.completed_at = time.time()
            job.error = None
            self._total_completed += 1
            log.info("Job %s completed successfully in %.3fs", job.job_id, job.completed_at - job.started_at)
        except Exception as exc:  # noqa: BLE001
            err_msg = f"{type(exc).__name__}: {str(exc)}"
            job.error_history.append(err_msg)
            job.error = err_msg
            job.retries_attempted += 1

            if job.retries_attempted <= job.max_retries:
                job.status = JobStatus.RETRYING
                self._total_retried += 1
                delay = (job.backoff_base ** (job.retries_attempted - 1))
                log.warning(
                    "Job %s failed (%s). Retrying in %.2fs (attempt %d/%d)",
                    job.job_id,
                    err_msg,
                    delay,
                    job.retries_attempted,
                    job.max_retries,
                )
                await asyncio.sleep(delay)
                await self._queue.put(job.job_id)
            else:
                job.status = JobStatus.FAILED
                job.completed_at = time.time()
                self._total_failed += 1
                self._dead_letter_queue.append(job)
                log.error(
                    "Job %s exhausted all %d retries. Moved to Dead-Letter Queue. Final error: %s",
                    job.job_id,
                    job.max_retries,
                    err_msg,
                )

    def stats(self) -> dict[str, Any]:
        """Summary metrics for monitoring & observability."""
        return {
            "queue_depth": self._queue.qsize(),
            "total_submitted": self._total_submitted,
            "total_completed": self._total_completed,
            "total_retried": self._total_retried,
            "total_failed": self._total_failed,
            "dead_letter_count": len(self._dead_letter_queue),
            "workers_active": len(self._workers),
        }


# Singleton job manager instance
job_manager = JobManager()
