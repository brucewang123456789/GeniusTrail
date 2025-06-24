import os
import pytest
from httpx import ASGITransport, AsyncClient

from api_server import app

API_TOKEN = "test-token"
os.environ["VELTRAX_API_TOKEN"] = API_TOKEN
HEADERS = {"Authorization": f"Bearer {API_TOKEN}"}


@pytest.mark.asyncio
async def test_ping() -> None:
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/ping")
    assert resp.status_code == 200
    assert resp.json() == {"pong": True}


@pytest.mark.asyncio
async def test_chat_basic() -> None:
    payload = {"prompt": "Hello", "history": [{"role": "user", "content": "Hi"}]}
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport, base_url="http://test", headers=HEADERS
    ) as client:
        resp = await client.post("/chat", json=payload)

    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data.get("response"), str)
    assert isinstance(data.get("used_cot"), bool)
    assert isinstance(data.get("duration_ms"), int)
