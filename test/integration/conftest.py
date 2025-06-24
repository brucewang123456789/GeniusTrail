import sys
from pathlib import Path
import os
import subprocess
import pytest

# Ensure project root on sys.path so veltraxor package can be found
sys.path.insert(0, str(Path(__file__).parents[2]))

# ────────────────────────────── NEW: disable auth for all integration tests ──
from veltraxor import app, verify_token


@pytest.fixture(autouse=True, scope="session")
def disable_auth():
    """
    Globally bypass token verification during integration tests.
    """
    app.dependency_overrides[verify_token] = lambda: None
    yield
    app.dependency_overrides.pop(verify_token, None)


# ───────────────────────────────────────── start optional services ───────────
@pytest.fixture(scope="session", autouse=True)
def start_services():
    """
    In CI (or when SKIP_SERVICE_START / IN_DOCKER set), assume services
    are provided by GitHub Actions. Locally, if no skip/CI/docker flag,
    bring up via docker-compose.test.yml.
    """
    skip = os.getenv("SKIP_SERVICE_START", "").lower() in {"1", "true", "yes"}
    is_ci = os.getenv("CI", "").lower() == "true" or os.getenv("GITHUB_ACTIONS", "").lower() == "true"
    in_docker = os.getenv("IN_DOCKER", "").lower() == "true"
    if skip or is_ci or in_docker:
        return

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
    except subprocess.CalledProcessError as exc:
        pytest.skip(f"docker-compose up failed ({exc}); skipping service startup")

