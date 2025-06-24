import os
import pytest
from fastapi.testclient import TestClient
from api_server import app
import llm_client

client = TestClient(app)
TOKEN = os.getenv("VELTRAX_API_TOKEN", "")


@pytest.mark.parametrize(
    "payload, expected_status",
    [
        ({"prompt": "", "history": []}, 400),
        ({"prompt": "a" * 10001, "history": []}, 400),
        ({"prompt": "normal prompt", "history": []}, 200),
    ],
)
def test_empty_or_too_long_prompt(payload, expected_status):
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    response = client.post("/chat", json=payload, headers=headers)
    assert response.status_code == expected_status


def test_missing_token():
    response = client.post("/chat", json={"prompt": "test", "history": []})
    assert response.status_code == 401


def test_invalid_token():
    response = client.post(
        "/chat",
        json={"prompt": "test", "history": []},
        headers={"Authorization": "Bearer invalid_token"},
    )
    assert response.status_code in (401, 403)


def test_upstream_timeout(monkeypatch):
    # Patch LLMClient.chat to simulate upstream timeout/error
    def fake_chat(self, messages):
        raise TimeoutError("Simulated upstream timeout")

    monkeypatch.setattr(llm_client.LLMClient, "chat", fake_chat)

    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    response = client.post(
        "/chat", json={"prompt": "test", "history": []}, headers=headers
    )
    assert response.status_code == 500
