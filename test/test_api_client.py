from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from api_server import app
from llm_client import LLMClient

# ------------------------------------------------------------------ constants
API_TOKEN = "test-token"
os.environ["VELTRAX_API_TOKEN"] = API_TOKEN  # ensure token for all tests
client = TestClient(app)


# ------------------------------------------------------------------ fixtures
@pytest.fixture(autouse=True)
def _inject_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guarantee env token for each test."""
    monkeypatch.setenv("VELTRAX_API_TOKEN", API_TOKEN)


# ------------------------------------------------------------------ tests
def test_ping_healthcheck() -> None:
    """The `/ping` endpoint must respond 200 with JSON body `{"pong": True}`."""
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"pong": True}


def test_chat_requires_auth() -> None:
    """Posting to /chat without Authorization header should return 401."""
    payload: Dict[str, Any] = {"prompt": "hi", "history": []}
    response = client.post("/chat", json=payload)
    assert response.status_code == 401


def test_chat_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock LLMClient.chat and verify /chat returns expected structure."""

    def fake_chat(self: LLMClient, messages: List[Dict[str, str]]) -> Dict[str, Any]:  # type: ignore[override]  # noqa: D401,E501
        return {"choices": [{"message": {"content": "Hello, world!"}}]}

    monkeypatch.setattr(LLMClient, "chat", fake_chat, raising=True)

    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    payload: Dict[str, Any] = {"prompt": "hi", "history": []}

    response = client.post("/chat", json=payload, headers=headers)
    assert response.status_code == 200

    data = response.json()
    assert data["response"] == "Hello, world!"
    assert isinstance(data["used_cot"], bool)
    assert isinstance(data["duration_ms"], int) and data["duration_ms"] >= 0
