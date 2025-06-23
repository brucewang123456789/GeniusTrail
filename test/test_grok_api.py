import os
import pytest
import respx
from httpx import Response, TimeoutException

from llm_client import LLMClient
from config import settings

# Ensure we use the configured API URL (default to this if not set)
API_URL = getattr(settings, "XAI_API_URL", "https://api.grok-3.ai/v1/chat")

@pytest.fixture(autouse=True)
def configure_environment(monkeypatch):
    """
    Configure environment variables and settings for stable testing.
    - Disable live calls by default.
    - Inject dummy API key.
    """
    monkeypatch.setenv("CI", "true")
    monkeypatch.setenv("RUN_LIVE_TESTS", "false")
    monkeypatch.setenv("VELTRAX_API_TOKEN", "dummy_token")
    # Override settings attributes if present
    monkeypatch.setattr(settings, "XAI_API_KEY", "dummy_token", raising=False)
    monkeypatch.setattr(settings, "XAI_API_URL", API_URL, raising=False)
    monkeypatch.setattr(settings, "VELTRAX_MODEL", "test-model", raising=False)

@pytest.fixture
def grok_client():
    """Instantiate a fresh LLMClient for each test."""
    return LLMClient()

@respx.mock
def test_timeout_returns_stub(grok_client):
    """
    When the HTTP client times out,
    chat() should catch it and return a stubbed response.
    """
    # Mock a network timeout on POST
    respx.post(API_URL).mock(side_effect=TimeoutException("request timed out"))
    result = grok_client.chat([{"role": "user", "content": "hello"}])

    # Validate stub structure
    assert isinstance(result, dict)
    choices = result.get("choices")
    assert isinstance(choices, list) and choices, "Expected non-empty choices list"
    content = choices[0].get("message", {}).get("content", "")
    assert content.startswith("[stub]"), f"Stub content must start with '[stub]', got {content!r}"

@respx.mock
@pytest.mark.parametrize("status_code", [400, 401, 429, 500, 503])
def test_http_error_status_returns_stub(grok_client, status_code):
    """
    If the API returns HTTP errors (4xx or 5xx),
    chat() should return a stubbed response as well.
    """
    respx.post(API_URL).mock(return_value=Response(status_code, json={"error": "failure"}))
    result = grok_client.chat([{"role": "user", "content": "test"}])

    assert isinstance(result, dict)
    choices = result.get("choices")
    assert choices and isinstance(choices, list)
    content = choices[0]["message"]["content"]
    assert content.startswith("[stub]")

@respx.mock
def test_successful_response_returns_payload(grok_client):
    """
    When the API returns 200 and valid JSON,
    chat() should return the exact payload.
    """
    payload = {"choices": [{"message": {"content": "All good"}}]}
    respx.post(API_URL).mock(return_value=Response(200, json=payload))
    result = grok_client.chat([{"role": "user", "content": "anything"}])

    assert result == payload

@pytest.mark.skipif(
    os.getenv("RUN_LIVE_TESTS", "false").lower() != "true",
    reason="Live tests are disabled outside dedicated environments"
)
def test_live_integration_chat():
    """
    A live integration test against the real Grok 3 API.
    Only runs when RUN_LIVE_TESTS=true and valid credentials are set.
    """
    client = LLMClient()
    response = client.chat([{"role": "user", "content": "live test"}])
    assert isinstance(response, dict)
    assert "choices" in response
