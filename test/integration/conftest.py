import subprocess
import time
import os
import pytest
from pathlib import Path

# Compose file at project root
COMPOSE = Path(__file__).parents[2] / 'docker-compose.test.yml'

@pytest.fixture(scope="session", autouse=True)
def start_services():
    subprocess.run(["docker-compose", "-f", str(COMPOSE), "up", "-d"], check=True)
    time.sleep(5)
    yield
    subprocess.run(["docker-compose", "-f", str(COMPOSE), "down", "-v"], check=True)

@pytest.fixture(scope="session")
def API_TOKEN():
    token = os.getenv("VELTRAX_API_TOKEN")
    if not token:
        pytest.skip("VELTRAX_API_TOKEN not set")
    return token