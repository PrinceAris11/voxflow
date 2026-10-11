"""test_api_v1.py — Integration tests for REST API v1 endpoints and background jobs."""

import time
from starlette.testclient import TestClient
from main import app

client = TestClient(app)


def test_api_v1_health():
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "uptime_seconds" in data
    assert "job_stats" in data
    assert "memory_stats" in data
    assert "X-Process-Time" in res.headers
    assert "X-Request-ID" in res.headers


def test_api_v1_refine_sync():
    payload = {"text": "um basically we are gonna ship today you know"}
    res = client.post("/api/v1/refine/sync", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["type"] == "transcript"
    assert "going to" in data["text_baseline"]
    assert "product" not in data["best_word"] or data["best_word"] == "ship"
    assert "memory" in data


def test_api_v1_job_submission_and_lifecycle():
    with TestClient(app) as test_client:
        payload = {
            "text": "so um like we gotta fix the database migration",
            "idempotency_key": "job-test-key-1"
        }
        res = test_client.post("/api/v1/jobs/refine", json=payload)
        assert res.status_code == 202
        job_info = res.json()
        assert "job_id" in job_info
        job_id = job_info["job_id"]
        assert job_info["status"] in ("pending", "running", "completed")

        # Idempotent re-submission returns the same job
        res_duplicate = test_client.post("/api/v1/jobs/refine", json=payload)
        assert res_duplicate.status_code == 202
        assert res_duplicate.json()["job_id"] == job_id

        # Poll status
        completed = False
        for _ in range(40):
            poll = test_client.get(f"/api/v1/jobs/{job_id}")
            assert poll.status_code == 200
            poll_data = poll.json()
            if poll_data["status"] == "completed":
                completed = True
                assert poll_data["result"] is not None
                break
            time.sleep(0.05)

        assert completed is True


def test_api_v1_cache_stats():
    res = client.get("/api/v1/cache/stats")
    assert res.status_code == 200
    data = res.json()
    assert "hits" in data
    assert "misses" in data
    assert "size" in data


def test_api_v1_validation_error_on_empty_text():
    res = client.post("/api/v1/refine/sync", json={"text": ""})
    assert res.status_code == 422
