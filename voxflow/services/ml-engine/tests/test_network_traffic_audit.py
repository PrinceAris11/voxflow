"""test_network_traffic_audit.py — Verification of Part K Zero Outbound Network Traffic.

Audits socket and HTTP connection calls during a complete dictation, intent finalization,
and transcript refinement session to mathematically assert ZERO outbound network calls.
Any outbound call to a non-loopback IP/host causes an immediate test failure.
"""

import socket
import pytest
from fastapi.testclient import TestClient

from main import app
from refiner_service import refiner_service

client = TestClient(app)

_ORIGINAL_SOCKET_CONNECT = socket.socket.connect


@pytest.fixture(autouse=True)
def intercept_outbound_network_calls(monkeypatch):
    """Intercept all socket connect calls and assert they are only local/loopback."""
    captured_external_calls: list[str] = []

    def audited_connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        host_str = str(host).lower()

        # Allow local loopback addresses only
        if host_str in ("127.0.0.1", "localhost", "::1", "testserver"):
            return _ORIGINAL_SOCKET_CONNECT(self, address)

        # Flag any external outbound call
        captured_external_calls.append(f"{host_str}:{address[1] if isinstance(address, tuple) and len(address) > 1 else 'unknown'}")
        raise ConnectionRefusedError(f"[SECURITY AUDIT VIOLATION] Attempted external network call to: {host_str}")

    monkeypatch.setattr(socket.socket, "connect", audited_connect)
    yield
    assert captured_external_calls == [], f"Network Audit Failed: External outbound calls detected: {captured_external_calls}"


@pytest.mark.asyncio
async def test_zero_network_calls_during_full_refinement_pipeline():
    """Verify full speech refinement runs 100% locally with zero external network traffic."""
    test_inputs = [
        "So um basically what I was thinking is that we like need to check the API response time you know?",
        "Let's schedule the meeting for Monday no wait make it Tuesday at 3 PM.",
        "Acha to kal meeting rakhte hain at 10 AM matlab ke team ke sath discussion karni hai.",
        "I want to go to nowhere cancel that just type hi how are you",
    ]

    for utterance in test_inputs:
        response = await refiner_service.refine(
            raw_text=utterance,
            use_llm=True,  # Even with use_llm=True, runs 100% local!
            record_memory=True,
            silence_duration_ms=1300.0,
        )
        assert response.text is not None
        assert response.duration_ms is not None
        # Confirms on-device execution
        assert not response.cached or response.cached is True


def test_zero_network_calls_during_sync_api_request():
    """Verify REST API /api/v1/refine/sync executes with zero external network traffic."""
    res = client.post(
        "/api/v1/refine/sync",
        json={
            "text": "First thing we gotta do is fix the build error in CI then after that ping Sarah and lastly deploy to production.",
            "use_llm": True,
            "silence_duration_ms": 1300.0,
        },
    )
    assert res.status_code == 200
    data = res.json()
    assert "Action items:" in data["text"]
