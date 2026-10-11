"""test_job_manager.py — Unit tests for the background job processing engine."""

import asyncio
import pytest
from job_manager import JobManager, JobStatus


@pytest.mark.asyncio
async def test_job_lifecycle_success():
    manager = JobManager(worker_count=1, max_retries=2, backoff_base=1.1)

    async def mock_processor(job):
        await asyncio.sleep(0.01)
        return {"processed": True, "text": job.raw_text.upper()}

    manager.set_processor(mock_processor)
    await manager.start()

    try:
        job = await manager.submit_job("hello world")
        assert job.status in (JobStatus.PENDING, JobStatus.RUNNING)

        # Wait for worker to complete
        for _ in range(30):
            if job.status == JobStatus.COMPLETED:
                break
            await asyncio.sleep(0.05)

        assert job.status == JobStatus.COMPLETED
        assert job.result == {"processed": True, "text": "HELLO WORLD"}
        assert job.completed_at is not None
        assert job.error is None
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_job_idempotency():
    manager = JobManager(worker_count=1)

    async def slow_processor(job):
        await asyncio.sleep(0.5)
        return "done"

    manager.set_processor(slow_processor)
    await manager.start()

    try:
        job1 = await manager.submit_job("repeatable utterance", idempotency_key="unique-idemp-1")
        job2 = await manager.submit_job("repeatable utterance", idempotency_key="unique-idemp-1")

        # Must return the exact same job ID
        assert job1.job_id == job2.job_id
    finally:
        await manager.shutdown()


@pytest.mark.asyncio
async def test_job_retry_and_dead_letter_queue():
    manager = JobManager(worker_count=1, max_retries=2, backoff_base=1.1)
    attempt_count = 0

    async def failing_processor(job):
        nonlocal attempt_count
        attempt_count += 1
        raise ValueError(f"Transient error #{attempt_count}")

    manager.set_processor(failing_processor)
    await manager.start()

    try:
        job = await manager.submit_job("will fail")

        # Wait for retries to exhaust
        for _ in range(50):
            if job.status == JobStatus.FAILED:
                break
            await asyncio.sleep(0.1)

        assert job.status == JobStatus.FAILED
        assert job.retries_attempted == 3  # Initial try + 2 retries
        assert "ValueError" in (job.error or "")

        # Check Dead Letter Queue
        dlq = manager.get_dead_letter_queue()
        assert len(dlq) >= 1
        assert dlq[-1].job_id == job.job_id
    finally:
        await manager.shutdown()
