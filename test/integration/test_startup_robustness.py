from fastapi.testclient import TestClient
from veltraxor import app

client = TestClient(app)


def test_ping_endpoint() -> None:
    """Service must respond to /ping."""
    resp = client.get("/ping")
    assert resp.status_code == 200
    assert resp.json() == {"pong": True}


def test_chat_unauthorized(monkeypatch) -> None:
    """Missing token ⇒ 401."""
    monkeypatch.delenv("VELTRAX_API_TOKEN", raising=False)
    resp = client.post("/chat", json={"prompt": "hi"})
    assert resp.status_code == 401


def test_chat_authorized(monkeypatch) -> None:
    """Correct token ⇒ 200."""
    monkeypatch.setenv("VELTRAX_API_TOKEN", "test-token")
    headers = {"Authorization": "Bearer test-token"}
    resp = client.post("/chat", json={"prompt": "hello"}, headers=headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "response" in body and body["duration_ms"] >= 0
