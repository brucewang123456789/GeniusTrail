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

    compose_file = Path(__file__).parents[2] / "docker-compose.test.yml"
    if not compose_file.exists():
        pytest.skip(f"docker-compose file not found at {compose_file}")
    try:
        subprocess.run(
            ["docker-compose", "-f", str(compose_file), "up", "-d"],
            check=True,
            cwd=str(compose_file.parent),
        )
    except FileNotFoundError:
        pytest.skip("docker-compose not installed; skipping service startup")
    except subprocess.CalledProcessError as e:
        pytest.skip(f"docker-compose up failed ({e}); skipping service startup")
