import os
import pytest
from httpx import AsyncClient, ASGITransport
from api_server import app

API_TOKEN = os.getenv("VELTRAX_API_TOKEN")            # optional in CI
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"} if API_TOKEN else {}

@pytest.mark.asyncio
async def test_ping():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ping")
    assert resp.status_code == 200
    assert resp.json() == {"pong": True}

@pytest.mark.asyncio
async def test_chat_basic():
    payload = {"prompt": "Health check", "history": []}
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", headers=HEADERS
    ) as client:
        resp = await client.post("/chat", json=payload)
    assert resp.status_code == 200, f"Unexpected status {resp.status_code}"
    data = resp.json()
    assert isinstance(data.get("response"), str)
    assert isinstance(data.get("used_cot"), bool)
    assert isinstance(data.get("duration_ms"), int)
