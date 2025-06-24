import os
import subprocess
import pytest
from pathlib import Path


@pytest.fixture(scope="session", autouse=True)
def start_services():
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

    # Local dev: bring up via docker-compose.test.yml
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
