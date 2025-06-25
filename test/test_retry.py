# test/test_retry.py

import asyncio
import json

import httpx
import pytest
import respx

from llm_client import LLMClient
from config import settings

@pytest.fixture(autouse=True)
def reset_stub_mode(monkeypatch):
    """
    Ensure each test runs in real mode unless explicitly mocked.
    """
    monkeypatch.setenv("CI", "false")
    settings.MOCK_LLM = False
    settings.XAI_API_KEY = "test-key"
    yield

@respx.mock
def test_chat_retries_until_success(respx_mock):
    """
    Simulate a transient network error on the first call and a successful 200 response on retry.
    Assert that chat() retries once and returns the correct content.
    """
    url = settings.XAI_API_URL

    # First request raises RequestError, second returns a valid JSON payload
    route = respx_mock.post(url).mock(
        side_effect=[
            httpx.RequestError("temporary network failure"),
            httpx.Response(
                200,
                json={
                    "id": "resp-123",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "Hello!"},
                            "finish_reason": "stop"
                        }
                    ],
                },
            ),
        ]
    )

    client = LLMClient()
    response = client.chat([{"role": "user", "content": "Hi"}])

    assert route.call_count == 2
    assert response["choices"][0]["message"]["content"] == "Hello!"

@respx.mock
def test_chat_fallback_to_stub_after_max_retries(respx_mock):
    """
    Simulate repeated timeouts exceeding max_retries.
    Assert that after exhausting retries, chat() falls back to stub mode.
    """
    url = settings.XAI_API_URL

    # All attempts raise TimeoutException (assume MAX_RETRIES = 2)
    route = respx_mock.post(url).mock(
        side_effect=[httpx.TimeoutException("timeout")] * 3
    )

    client = LLMClient()
    response = client.chat([{"role": "user", "content": "Test stub"}])

    # Expect 2 real attempts, then stub fallback
    assert route.call_count == settings.MAX_RETRIES
    assert response["choices"][0]["message"]["content"].startswith("[stub]")

@pytest.mark.asyncio
async def test_stream_chat_fallback_stub_on_error(monkeypatch):
    """
    Simulate a connection failure in the async stream method.
    Assert that stream_chat() yields only the stub stream output.
    """
    # Override AsyncClient.stream to return a context manager whose __aenter__ raises ConnectError
    def fake_stream(self, *args, **kwargs):
        class BrokenStreamCtx:
            async def __aenter__(self_inner):
                raise httpx.ConnectError("connect failed")
            async def __aexit__(self_inner, exc_type, exc, tb):
                pass
        return BrokenStreamCtx()

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)

    client = LLMClient()
    collected = [chunk async for chunk in client.stream_chat([{"role": "user", "content": "Async"}])]

    assert collected == ["[stub-stream] Async"]
