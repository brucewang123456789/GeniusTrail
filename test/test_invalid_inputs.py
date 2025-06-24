import os
from typing import Dict

import pytest
from fastapi.testclient import TestClient

from api_server import app
import llm_client

API_TOKEN = "test-token"
os.environ["VELTRAX_API_TOKEN"] = API_TOKEN

client = TestClient(app)


@pytest.mark.parametrize(
    "payload, expected_status",
    [
        ({"prompt": "", "history": []}, 400),
        ({"prompt": "a" * 10001, "history": []}, 400),
        ({"prompt": "normal prompt", "history": []}, 200),
    ],
)
def test_empty_or_too_long_prompt(
    payload: Dict[str, str], expected_status: int
) -> None:
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = client.post("/chat", json=payload, headers=headers)
    assert response.status_code == expected_status


def test_missing_token() -> None:
    response = client.post("/chat", json={"prompt": "test", "history": []})
    assert response.status_code == 401


def test_invalid_token() -> None:
    response = client.post(
        "/chat",
        json={"prompt": "test", "history": []},
        headers={"Authorization": "Bearer invalid_token"},
    )
    assert response.status_code == 403


def test_upstream_timeout(monkeypatch) -> None:
    """Simulate upstream timeout/error and expect 500."""

    def fake_chat(self, messages):  # type: ignore[no-self-use]
        raise TimeoutError("Simulated upstream timeout")

    monkeypatch.setattr(llm_client.LLMClient, "chat", fake_chat)

    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = client.post(
        "/chat", json={"prompt": "test", "history": []}, headers=headers
    )
    assert response.status_code == 500
