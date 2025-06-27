import os
import pytest
import respx
from httpx import Response, TimeoutException
from llm_client import LLMClient
from config import settings

API_URL = "https://api.grok-3.ai/v1/chat"

@pytest.fixture(autouse=True)
def configure_environment(monkeypatch):
    # disable stub mode and live tests by default
    monkeypatch.setenv("CI", "false")
    monkeypatch.setenv("RUN_LIVE_TESTS", "false")
    monkeypatch.setattr(settings, "XAI_API_KEY", "dummy_token", raising=False)
    monkeypatch.setattr(settings, "XAI_API_URL", API_URL, raising=False)
    monkeypatch.setattr(settings, "VELTRAX_MODEL", "test-model", raising=False)

@pytest.fixture
def grok_client():
    return LLMClient()

@respx.mock
def test_timeout_returns_stub(grok_client):
    respx.post(API_URL).mock(side_effect=TimeoutException("timeout"))
    result = grok_client.chat([{"role": "user", "content": "hello"}])
    assert isinstance(result, dict)
    assert "choices" in result
    assert result["choices"][0]["message"]["content"].startswith("[stub]")

@respx.mock
@pytest.mark.parametrize("status_code", [401, 429, 500, 503])
def test_http_errors_return_stub(grok_client, status_code):
    respx.post(API_URL).mock(return_value=Response(status_code, json={"error": "failure"}))
    result = grok_client.chat([{"role": "user", "content": "test"}])
    assert isinstance(result, dict)
    assert result["choices"][0]["message"]["content"].startswith("[stub]")

@respx.mock
def test_successful_response_returns_json(grok_client):
    payload = {"choices": [{"message": {"content": "OK"}}]}
    respx.post(API_URL).mock(return_value=Response(200, json=payload))
    result = grok_client.chat([{"role": "user", "content": "test"}])
    assert result == payload

@pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS", "false").lower() != "true",
    reason="Skipping live integration test in static-only environment"
)
def test_live_integration_chat():
    client = LLMClient()
    response = client.chat([{"role": "user", "content": "hello"}])
    assert isinstance(response, dict)
    assert "choices" in response
