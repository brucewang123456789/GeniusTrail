from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from api_server import app  # Ensure api_server.py exposes `app = FastAPI(...)`
from llm_client import LLMClient

client = TestClient(app)


@pytest.fixture(autouse=True)
def inject_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Ensure VELTRAX_API_TOKEN is set during tests.
    """
    monkeypatch.setenv(
        "VELTRAX_API_TOKEN", os.getenv("VELTRAX_API_TOKEN", "dummy_token")
    )


def test_ping_healthcheck() -> None:
    """
    The `/ping` endpoint should return status 200 with JSON body {"pong": True}.
    """
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"pong": True}


def test_chat_requires_auth() -> None:
    """
    Posting to /chat without Authorization header should be rejected.
    """
    payload: Dict[str, Any] = {"prompt": "hi", "history": []}
    response = client.post("/chat", json=payload)
    assert response.status_code in (401, 403)


def test_chat_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Mock LLMClient.chat to return the structure matching production expectation,
    then verify /chat endpoint returns expected ChatResponse fields.
    """

    def fake_chat(self: LLMClient, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        return {"choices": [{"message": {"content": "Hello, world!"}}]}

    # Ensure token is in environment and patch the chat method
    monkeypatch.setenv(
        "VELTRAX_API_TOKEN", os.getenv("VELTRAX_API_TOKEN", "dummy_token")
    )
    monkeypatch.setattr(LLMClient, "chat", fake_chat, raising=True)

    headers = {"Authorization": f"Bearer {os.environ['VELTRAX_API_TOKEN']}"}
    payload: Dict[str, Any] = {"prompt": "hi", "history": []}

    response = client.post("/chat", json=payload, headers=headers)
    assert response.status_code == 200

    data = response.json()
    # Production ChatResponse is expected to include:
    #   - "response": the text returned by fake_chat
    #   - "used_cot": a boolean
    #   - "duration_ms": a non-negative integer
    assert "response" in data and data["response"] == "Hello, world!"
    assert "used_cot" in data and isinstance(data["used_cot"], bool)
    assert (
        "duration_ms" in data
        and isinstance(data["duration_ms"], int)
        and data["duration_ms"] >= 0
    )
