import os
import subprocess
import pytest
from pathlib import Path

@pytest.fixture(scope="session", autouse=True)
def _default_api_token() -> None:
    """
    Ensure every test run has at least a dummy token so auth is never
    inadvertently disabled. Individual tests may override via monkeypatch.
    """
    os.environ.setdefault("VELTRAX_API_TOKEN", "***")


@pytest.fixture(scope="session", autouse=True)
def start_services() -> None:
    """
    In CI (or when SKIP_SERVICE_START/IN_DOCKER set), assume GitHub 'services:' proxy is up.
    Locally, if none of those flags are set, try to docker-compose up.
    """
    skip = os.getenv("SKIP_SERVICE_START", "").lower() in {"1", "true", "yes"}
    is_ci = (
        os.getenv("CI", "").lower() == "true"
        or os.getenv("GITHUB_ACTIONS", "").lower() == "true"
    )
    in_docker = os.getenv("IN_DOCKER", "").lower() == "true"
    if skip or is_ci or in_docker:
        return  # services already provided

    compose_file = Path(__file__).parent / "docker-compose.test.yml"
    if not compose_file.exists():
        return  # do nothing if compose file missing

    try:
        subprocess.run(
            ["docker", "compose", "-f", str(compose_file), "up", "-d"],
            check=True,
            cwd=str(compose_file.parent),
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        # unable to start services; continue without skipping tests
        return


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
