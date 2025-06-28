import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.dependencies import redis_client, llm_client
from llm_client import LLMClient
import httpx

client = TestClient(app)


def test_readiness_redis_failure(monkeypatch):
    """If Redis ping raises, /readiness should return 503."""
    async def fail_ping():
        raise ConnectionError("simulated Redis failure")
    monkeypatch.setattr(redis_client, "ping", fail_ping)
    resp = client.get("/readiness")
    assert resp.status_code == 503


def test_readiness_llm_failure(monkeypatch):
    """If LLM client chat raises, /readiness should return 503."""
    def fail_chat(*args, **kwargs):
        raise RuntimeError("simulated LLM failure")
    monkeypatch.setattr(llm_client, "chat", fail_chat)
    resp = client.get("/readiness")
    assert resp.status_code == 503


def test_llm_client_http_failure_stub(monkeypatch):
    """
    If httpx.post always errors and MAX_RETRIES exhausted,
    LLMClient.chat must fall back to stub implementation.
    """
    client_real = LLMClient(model="test-model", base_url="http://example.com")
    client_real._stub_mode = False

    def always_fail_post(*args, **kwargs):
        raise httpx.ConnectError("simulated connect error")
    monkeypatch.setattr(httpx, "post", always_fail_post)

    result = client_real.chat([{"role": "user", "content": "hello"}])
    assert result["choices"][0]["message"]["content"] == "[stub] hello"
