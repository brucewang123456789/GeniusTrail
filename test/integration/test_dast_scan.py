import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_path_traversal_returns_404():
    """
    Attempt a directory-traversal URL; server must not serve internal files.
    """
    resp = client.get("/static/../config.py")
    assert resp.status_code in (400, 404), "Directory traversal should be blocked"


def test_crlf_injection_not_reflected():
    """
    Inject CRLF into a header value; server must sanitize and not reflect new header.
    """
    malicious = "value\r\nInjected-Header: evil=1"
    headers = {"X-Test": malicious}
    resp = client.get("/liveness", headers=headers)
    assert "Injected-Header" not in resp.headers
