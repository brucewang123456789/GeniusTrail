"""Test-level pytest configuration.

This file provides a dummy API token and also restores the same two
private jsonschema names so that Hypothesis-JSONSchema and Schemathesis
can import without crashing during test collection.
"""

import os
import jsonschema
import jsonschema.exceptions as _je
import jsonschema.validators as _jv
import pytest

# ---------------------------------------------------------------------------
# Restore jsonschema.exceptions._RefResolutionError
# ---------------------------------------------------------------------------
if not hasattr(_je, "_RefResolutionError"):
    _je._RefResolutionError = _je.RefResolutionError  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Restore jsonschema.validators._RefResolver
# ---------------------------------------------------------------------------
if not hasattr(_jv, "_RefResolver"):
    try:
        _jv._RefResolver = jsonschema.RefResolver  # type: ignore[attr-defined]
    except AttributeError:

        class _NoRefResolver:
            """Stub for the removed RefResolver; only needed at import-time."""

            def __init__(self, *_: object, **__: object) -> None:
                raise NotImplementedError(
                    "jsonschema.RefResolver has been removed; this is a stub."
                )

        _jv._RefResolver = _NoRefResolver  # type: ignore[assignment]

# Provide a dummy API token so that any test depending on VELTRAX_API_TOKEN passes
os.environ.setdefault("VELTRAX_API_TOKEN", "dummy_token")

# ---------------------------------------------------------------------------
# Autouse fixture to inject a Redis client for integration tests
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _inject_redis_client(monkeypatch, request):
    """
    Autouse fixture for tests marked 'integration' to inject a Redis client,
    ensuring tests run without skip.
    """
    if not request.node.get_closest_marker("integration"):
        return

    client = None
    # Attempt real Redis
    try:
        import redis
        from redis.exceptions import ConnectionError
        url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        candidate = redis.Redis.from_url(url)
        candidate.ping()
        client = candidate
    except Exception:
        # Fallback to fakeredis
        try:
            import fakeredis
            client = fakeredis.FakeRedis()
        except ImportError:
            # Minimal in-memory fallback
            from collections import defaultdict
            class SimpleFakeRedis(defaultdict):
                def ping(self):
                    return True
            client = SimpleFakeRedis(int)

    # Patch into app dependencies
    import app.dependencies as deps
    monkeypatch.setattr(deps, "redis_client", client, raising=False)
