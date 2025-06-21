# tests/test_api.py

import os
import pytest
from httpx import AsyncClient, ASGITransport
from api_server import app   # 确保指向定义 FastAPI app 的模块

TOKEN = os.getenv("VELTRAX_API_TOKEN")
if not TOKEN:
    raise RuntimeError("Environment variable VELTRAX_API_TOKEN is not set")

@pytest.mark.asyncio
async def test_ping():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"pong": True}

@pytest.mark.asyncio
async def test_chat_basic():
    headers = {"Authorization": f"Bearer {TOKEN}"}
    payload = {"prompt": "Health check", "history": []}
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test", headers=headers) as client:
        response = await client.post("/chat", json=payload)
    assert response.status_code == 200, f"Unexpected status code: {response.status_code}"
    data = response.json()
    assert isinstance(data.get("response"), str), "'response' must be a string"
    assert isinstance(data.get("used_cot"), bool), "'used_cot' must be a boolean"
    assert isinstance(data.get("duration_ms"), int), "'duration_ms' must be an integer"