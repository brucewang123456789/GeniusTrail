# File: tests/test_synthetic_monitoring.py

import pytest
from fastapi.testclient import TestClient
from starlette import status

from app.main import app

client = TestClient(app)


class FailingLLM:
    async def chat(self, *args, **kwargs):
        raise RuntimeError("LLM timeout")


@pytest.mark.asyncio
async def test_synthetic_readiness_then_chat(monkeypatch):
    """
    Simulate the scenario of LLM service timeout:
    - The readiness probe reports 503 first
    - The /chat_monitor interface is downgraded to the default "service busy" prompt
    - It can work normally after recovery
    """
    from app.dependencies import llm_client

    # Stage 1: LLM failure
    monkeypatch.setattr(llm_client, "chat", FailingLLM().chat)

    # readiness should be 503 when LLM times out
    resp_ready = client.get("/readiness")
    assert resp_ready.status_code == status.HTTP_503_SERVICE_UNAVAILABLE

    # /chat_monitor must return 200 + {"reply": "..."} on degradation
    resp_chat = client.post("/chat_monitor", json={"message": "Hello"})
    assert resp_chat.status_code == status.HTTP_200_OK
    assert "service busy" in resp_chat.json().get("reply", "").lower()

    # Phase 2: restore LLM client to real implementation
    from app.dependencies import LLMClient  # real implementation

    real_llm = LLMClient()
    monkeypatch.setattr(llm_client, "chat", real_llm.chat)

    # now readiness should be 200
    resp_ready2 = client.get("/readiness")
    assert resp_ready2.status_code == status.HTTP_200_OK

    # and /chat should return 200 + {"response": "...", ...}
    resp_chat2 = client.post("/chat", json={"prompt": "Hello again", "history": []})
    assert resp_chat2.status_code == status.HTTP_200_OK

    # check new response schema: 'response' field exists
    assert "response" in resp_chat2.json()
