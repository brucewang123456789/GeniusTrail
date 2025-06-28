"""Root-level pytest configuration with targeted Redis fallback."""

from __future__ import annotations

import importlib
import os

import jsonschema
import jsonschema.exceptions as _je
import jsonschema.validators as _jv
import pytest

# ---------------------------------------------------------------------------
#  jsonschema backward-compat shim  (unchanged)
# ---------------------------------------------------------------------------
if not hasattr(_je, "_RefResolutionError"):
    _je._RefResolutionError = _je.RefResolutionError  # type: ignore[attr-defined]

if not hasattr(_jv, "_RefResolver"):
    try:
        _jv._RefResolver = jsonschema.RefResolver  # type: ignore[attr-defined]
    except AttributeError:

        class _NoRefResolver:
            def __init__(self, *_: object, **__: object) -> None:
                raise NotImplementedError

        _jv._RefResolver = _NoRefResolver  # type: ignore[assignment]


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: mark tests as integration tests requiring external services",
    )


@pytest.fixture(autouse=True)
def _redis_or_fakeredis(monkeypatch, request):
    """
    For any test in test/integration/ marked 'integration',
    ensure Redis-compatible client is available; otherwise skip.
    """
    marker = request.node.get_closest_marker("integration")
    if not marker:
        return

    path = request.node.fspath.strpath.replace("\\", "/")
    if "/test/integration/" not in path:
        # ignore other integration-marked tests
        return

    import redis
    from redis.exceptions import ConnectionError

    url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    client = redis.Redis.from_url(url)
    try:
        client.ping()
        _patch_if_present(monkeypatch, "app.dependencies", "redis_client", client)
        return
    except ConnectionError:
        pass  # fallback to fakeredis

    spec = importlib.util.find_spec("fakeredis")
    if spec is None:
        pytest.skip(
            f"Cannot connect to Redis at {url} and fakeredis not installed; "
            "skipping integration test"
        )

    import fakeredis  # type: ignore

    fake_client = fakeredis.FakeRedis()
    _patch_if_present(monkeypatch, "app.dependencies", "redis_client", fake_client)


def _patch_if_present(monkeypatch, module_name: str, attr: str, value):
    """Helper: monkey-patch attr on module if module is importable."""
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        return
    if hasattr(mod, attr):
        monkeypatch.setattr(mod, attr, value, raising=False)
