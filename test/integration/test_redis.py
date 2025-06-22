import os
import pytest
import redis

"""
Integration test for Redis: skip if the service is not reachable
(i.e., local dev without Redis; CI provides Redis via GitHub Actions services).
"""


@pytest.mark.integration
def test_redis_set_get():
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", 6379))
    try:
        r = redis.Redis(host=host, port=port, db=0)
        r.set("ping", "pong")
        assert r.get("ping") == b"pong"
    except redis.exceptions.ConnectionError:
        pytest.skip("Redis service not available; skipping Redis integration test")
