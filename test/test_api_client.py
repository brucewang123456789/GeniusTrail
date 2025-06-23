"""
FastAPI TestClient integration tests.

These tests exercise the API routes without starting an external Uvicorn
process, so they are fast and reliable inside GitHub Actions.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

# Import the FastAPI app instance
from api_server import app  # make sure api_server.py exposes `app = FastAPI(...)`

client = TestClient(app)


@pytest.fixture(autouse=True)
def _inject_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Ensure required environment variables exist during tests.

    The CI runner injects secrets, but when you run pytest locally those
    variables may be absent.
    """
    monkeypatch.setenv("VELTRAX_API_TOKEN", os.getenv("VELTRAX_API_TOKEN", "dummy_token"))


def test_ping_healthcheck() -> None:
    """`/ping` should respond 200 and JSON body."""
    response = client.get("/ping")
    assert response.status_code == 200
    # Adjust field names to match your implementation
    assert response.json() == {"status": "ok"}


def test_chat_requires_auth() -> None:
    """Posting to /chat without Authorization header should be rejected."""
    payload: Dict[str, Any] = {"prompt": "hi", "history": []}
    response = client.post("/chat", json=payload)
    assert response.status_code in (401, 403)


def test_chat_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock LLM call and verify happy path."""
    # Patch LLMClient.chat to avoid external API call
    from llm_client import LLMClient

    def _fake_chat(self: LLMClient, messages: List[Dict[str, str]]) -> Dict[str, str]:
        return {"reply": "Hello, world!"}

    monkeypatch.setattr(LLMClient, "chat", _fake_chat, raising=True)

    headers = {"Authorization": f"Bearer {os.environ['VELTRAX_API_TOKEN']}"}
    payload: Dict[str, Any] = {"prompt": "hi", "history": []}
    response = client.post("/chat", json=payload, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert data == {"reply": "Hello, world!"}
