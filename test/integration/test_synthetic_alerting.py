import re
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.dependencies import redis_client

client = TestClient(app)


def test_error_counter_increments_on_readiness_failure(monkeypatch):
    """
    After a forced readiness failure, some errors counter for /readiness
    must be ≥1 in /metrics output.
    """

    async def fail_ping():
        raise ConnectionError("simulated Redis down")

    monkeypatch.setattr(redis_client, "ping", fail_ping)

    client.get("/readiness")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    text = resp.text

    # Any line with /readiness and a number ≥1
    m = re.search(r".*/readiness[^}\n]*\}\s*([1-9]\d*)", text)
    assert m, "No readiness error metric with count ≥1 was found"
    assert int(m.group(1)) >= 1


def test_request_counter_for_metrics_endpoint():
    """
    A successful GET /metrics should show at least 2 requests to /metrics.
    """
    client.get("/metrics")
    resp2 = client.get("/metrics")
    assert resp2.status_code == 200
    text = resp2.text

    # Any line with /metrics and a number ≥2
    m = re.search(r".*/metrics[^}\n]*\}\s*([2-9]\d*)", text)
    assert m, "No metrics request count ≥2 was found"
    assert int(m.group(1)) >= 2
