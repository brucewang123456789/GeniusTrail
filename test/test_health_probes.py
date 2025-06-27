# File: C:\Chatbot 8000\test\test_health_probes.py

import pytest
from fastapi.testclient import TestClient
from app.main import app
from starlette import status

client = TestClient(app)


class DummyRedis:
    async def ping(self):
        return True


class FailingRedis:
    async def ping(self):
        raise ConnectionError("Redis down")


@pytest.mark.asyncio
async def test_liveness_endpoint_up():
    """Liveness probe should always return 200 if app process is alive."""
    response = client.get("/liveness")
    assert response.status_code == status.HTTP_200_OK


@pytest.mark.asyncio
async def test_readiness_endpoint_all_dependencies_ok(monkeypatch):
    """Readiness returns 200 when Redis ping succeeds."""
    # Replace the actual redis client
    from app.dependencies import redis_client

    monkeypatch.setattr(redis_client, "ping", DummyRedis().ping)

    response = client.get("/readiness")
    assert response.status_code == status.HTTP_200_OK


@pytest.mark.asyncio
async def test_readiness_endpoint_redis_down(monkeypatch):
    """Readiness returns 503 when Redis ping fails."""
    from app.dependencies import redis_client

    monkeypatch.setattr(redis_client, "ping", FailingRedis().ping)

    response = client.get("/readiness")
    assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
