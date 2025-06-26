# File: C:\Chatbot 8000\tests\test_metrics_endpoint.py

import re
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_metrics_contains_core_counters():
    """
    /metrics should output Prometheus format metrics, including at least:
    - http_requests_total
    - http_request_errors_total
    - external_redis_latency_seconds
    """
    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text

    # Core indicator existence check
    core_metrics = [
        r"^http_requests_total\{.*\} \d+",
        r"^http_request_errors_total\{.*\} \d+",
        r"^external_redis_latency_seconds_bucket\{.*\} .*",
    ]
    for pattern in core_metrics:
        assert re.search(pattern, text, re.MULTILINE), f"Missing metric: {pattern}"
