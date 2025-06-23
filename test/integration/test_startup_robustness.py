import os
import sys
import types
import pytest
from fastapi.testclient import TestClient

# Stub out optional_client and db_client so production imports succeed without errors
optional_client_stub = types.ModuleType("optional_client")
def initialize_stub(*args, **kwargs):
    return True
optional_client_stub.initialize = initialize_stub
sys.modules["optional_client"] = optional_client_stub

db_client_stub = types.ModuleType("db_client")
def connect_stub(*args, **kwargs):
    return True
db_client_stub.connect = connect_stub
sys.modules["db_client"] = db_client_stub

from veltraxor import app
from config import get_settings

client = TestClient(app)

def test_ping_endpoint():
    """
    Basic liveness check: service starts and /ping must respond with 200.
    """
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"pong": True}

def test_chat_unauthorized(monkeypatch):
    """
    /chat requires a valid token; without it should return 401.
    """
    monkeypatch.delenv("VELTRAX_API_TOKEN", raising=False)
    response = client.post("/chat", json={"prompt": "hi"})
    assert response.status_code == 401

def test_chat_authorized(monkeypatch):
    """
    With correct token in environment and header, /chat returns 200 and includes expected fields.
    """
    monkeypatch.setenv("VELTRAX_API_TOKEN", "test-token")
    headers = {"Authorization": "Bearer test-token"}
    response = client.post("/chat", json={"prompt": "hello"}, headers=headers)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert isinstance(data["duration_ms"], (int, float))
    assert data["duration_ms"] >= 0

def test_optional_service_unavailable(monkeypatch):
    """
    Simulate optional external service initialization failure:
    service should still start, and /health reports external_service=False.
    """
    monkeypatch.setenv("VELTRAX_API_TOKEN", "test-token")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/db")

    oc = pytest.importorskip("optional_client")
    monkeypatch.setattr(oc, "initialize", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("init fail")))

    _ = get_settings()  # trigger initialization
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body.get("external_service") is False

def test_required_service_unavailable(monkeypatch):
    """
    Simulate required dependency failure: startup should fail fast.
    """
    monkeypatch.setenv("VELTRAX_API_TOKEN", "test-token")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")

    dbc = pytest.importorskip("db_client")
    monkeypatch.setattr(dbc, "connect", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("db down")))

    with pytest.raises(Exception):
        get_settings()

def test_env_config_variations(monkeypatch):
    """
    Startup with different environment configurations, e.g., disabling optional feature.
    """
    monkeypatch.setenv("VELTRAX_API_TOKEN", "test-token")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/db")
    monkeypatch.setenv("ENABLE_FEATURE_X", "false")

    _ = get_settings()
    response = client.get("/health")
    body = response.json()
    assert body.get("feature_x_enabled") is False

def test_malformed_config(monkeypatch):
    """
    Startup with malformed environment values should fail.
    """
    monkeypatch.setenv("VELTRAX_API_TOKEN", "test-token")
    monkeypatch.setenv("REDIS_URL", "not-a-valid-url")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@localhost:5432/db")

    with pytest.raises(Exception):
        get_settings()
