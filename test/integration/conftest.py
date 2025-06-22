import os
import subprocess
import pytest
from pathlib import Path

"""
conftest.py for integration tests: ensure services are available.

In local development (outside Docker/CI), if SKIP_SERVICE_START is not set,
attempt to run docker-compose to start Redis/Postgres.

In CI or inside Docker container, set CI=true or IN_DOCKER=true or SKIP_SERVICE_START=true
so this fixture does nothing, assuming services are started externally (e.g., GitHub Actions services).
"""

@pytest.fixture(scope="session", autouse=True)
def start_services():
    skip_flag = os.getenv("SKIP_SERVICE_START", "").lower() in {"1", "true", "yes"}
    is_ci = os.getenv("CI", "").lower() == "true"
    in_docker = os.getenv("IN_DOCKER", "").lower() == "true"
    if skip_flag or is_ci or in_docker:
        # Services (Redis/Postgres) should already be running (GitHub Actions services or external)
        return

    # Local development: attempt docker-compose up
    # Assume a file docker-compose.test.yml is located two levels up: project root/docker-compose.test.yml
    compose_file = Path(__file__).parents[2] / "docker-compose.test.yml"
    if not compose_file.exists():
        pytest.skip(f"docker-compose file not found at {compose_file}; skipping service startup")
    try:
        subprocess.run(
            ["docker-compose", "-f", str(compose_file), "up", "-d"],
            check=True,
            cwd=str(compose_file.parent),
        )
    except FileNotFoundError:
        pytest.skip("docker-compose not available; skipping service startup")
    except subprocess.CalledProcessError as e:
        pytest.skip(f"docker-compose up failed ({e}); skipping service startup")

    # Optionally wait until services are healthy; tests may retry connections themselves.
    # No teardown here; user can run "docker-compose down" manually if desired.
