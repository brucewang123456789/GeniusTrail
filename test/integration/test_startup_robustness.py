from __future__ import annotations

import os
from typing import Dict, List

from fastapi.testclient import TestClient

from veltraxor import app

API_TOKEN = "test-token"
client = TestClient(app)


def test_ping_endpoint() -> None:
    """/ping must respond 200."""
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert resp.json() == {"pong": True}


def test_chat_unauthorized(monkeypatch) -> None:
    """/chat without token should return 401."""
    monkeypatch.delenv("VELTRAX_API_TOKEN", raising=False)
    resp = client.post("/chat", json={"prompt": "hi"})
    assert resp.status_code == 401


def test_chat_authorized(monkeypatch) -> None:
    """With correct token, /chat returns 200."""
    monkeypatch.setenv("VELTRAX_API_TOKEN", API_TOKEN)
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    resp = client.post("/chat", json={"prompt": "hello"}, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "response" in body
    assert isinstance(body["duration_ms"], (int, float))
    assert body["duration_ms"] >= 0
