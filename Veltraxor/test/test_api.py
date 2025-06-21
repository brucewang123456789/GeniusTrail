# tests/test_api.py

import os
import pytest
from httpx import AsyncClient

BASE_URL = os.getenv("CHAT_API_URL", "http://127.0.0.1:8000")
TOKEN = os.getenv("VELTRAX_API_TOKEN")
if not TOKEN:
    raise RuntimeError("Environment variable VELTRAX_API_TOKEN is not set")

@pytest.mark.asyncio
async def test_ping():
    async with AsyncClient(base_url=BASE_URL) as client:
        response = await client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"pong": True}

@pytest.mark.asyncio
async def test_chat_basic():
    headers = {"Authorization": f"Bearer {TOKEN}"}
    payload = {"prompt": "Health check", "history": []}

    async with AsyncClient(base_url=BASE_URL, headers=headers, timeout=10.0) as client:
        response = await client.post("/chat", json=payload)

    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    data = response.json()
    assert isinstance(data.get("response"), str), "‘response’ must be a string"
    assert isinstance(data.get("used_cot"), bool),  "‘used_cot’ must be a boolean"
    assert isinstance(data.get("duration_ms"), int), "‘duration_ms’ must be an integer"