"""test_personalization.py — Verification of Part H Local Personalization & Adaptation.

Tests:
1. Recording correction feedback triples and edit rate calculation.
2. Vocabulary extraction and hotword addition for ASR decoder biasing.
3. Personalization metrics reporting and readiness threshold gating.
4. API endpoints for feedback logging and vocabulary queries.
"""

from fastapi.testclient import TestClient
import pytest

from main import app
from personalization_service import personalization_service

client = TestClient(app)


def test_personalization_record_correction():
    raw = "let's deploy to k8s cluster"
    output = "Let's deploy to k8s cluster."
    edited = "Let's deploy to Kubernetes cluster."

    feedback = personalization_service.record_correction(
        raw=raw,
        pipeline_output=output,
        user_edited=edited,
        app_context="VSCode",
    )

    assert feedback.character_distance > 0
    assert 0.0 < feedback.edit_rate <= 1.0
    # Auto-extracted capitalized term
    assert "Kubernetes" in personalization_service.vocabulary


def test_personalization_add_vocabulary():
    personalization_service.add_vocabulary_term("FastAPI")
    assert "FastAPI" in personalization_service.vocabulary


def test_personalization_metrics_summary():
    metrics = personalization_service.get_metrics_summary()
    assert metrics["total_corrections"] >= 1
    assert "average_edit_rate" in metrics
    assert "vocabulary_count" in metrics


def test_personalization_api_routes():
    # 1. Post feedback
    res = client.post(
        "/api/v1/personalization/feedback",
        json={
            "raw": "i need docker compose",
            "pipeline_output": "I need docker compose.",
            "user_edited": "I need Docker Compose.",
            "app_context": "Terminal",
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert data["logged"] is True

    # 2. Get metrics
    res_m = client.get("/api/v1/personalization/metrics")
    assert res_m.status_code == 200
    assert res_m.json()["total_corrections"] >= 1

    # 3. Add vocabulary
    res_v = client.post("/api/v1/personalization/vocabulary", json={"term": "OpenTelemetry"})
    assert res_v.status_code == 200

    # 4. Get vocabulary
    res_g = client.get("/api/v1/personalization/vocabulary")
    assert res_g.status_code == 200
    assert "OpenTelemetry" in res_g.json()["vocabulary"]
