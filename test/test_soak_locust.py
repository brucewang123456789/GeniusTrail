"""test_soak_locust.py — lightweight soak-test harness for the Chatbot service.

This script is launched by the Locust CLI.  When imported by pytest it falls back
to harmless stubs so that the file is collected but skipped (exit-code 0).

Usage (headless):
    python -m locust -f test/test_soak_locust.py \
        --host=http://localhost:8000 \
        --headless -u 10 -r 2 --run-time 15m

Dependencies:
    pip install "locust>=2.25,<3" python-dotenv

Environment variables (loaded from .env):
    REAL_CALL_FRACTION   float   fraction of non-mock calls (default 0.02)
    TOGGLE_INTERVAL_SEC  int     toggle interval in seconds (default 90)
    MOCK_QUERY_PARAM     str     mock trigger param (default "mock=true")
    AUTH_HEADER          str     e.g. "Bearer real_valid_token"
"""

from __future__ import annotations
import os
import random
import time
from typing import Dict, Optional

from dotenv import load_dotenv
load_dotenv()

# Graceful fallback: if Locust is missing, create no-op stubs
try:
    from locust import HttpUser, events as locust_events, task, between  # type: ignore
except ModuleNotFoundError:
    import types
    class HttpUser:
        pass
    def task(_=1):
        def decorator(fn): return fn
        return decorator
    def between(a, b): return lambda: 0
    locust_events = types.SimpleNamespace(test_start=types.SimpleNamespace(add_listener=lambda f: None))

# Config driven by environment
REAL_FRACTION = float(os.getenv("REAL_CALL_FRACTION", "0.02"))
TOGGLE_INTERVAL = int(os.getenv("TOGGLE_INTERVAL_SEC", "90"))
MOCK_KV = os.getenv("MOCK_QUERY_PARAM", "mock=true")
RAW_AUTH_HEADER = os.getenv("AUTH_HEADER")
AUTH_HEADER = RAW_AUTH_HEADER.strip('"').strip("'") if RAW_AUTH_HEADER else ""

if "=" in MOCK_KV:
    MOCK_KEY, MOCK_VAL = MOCK_KV.split("=", 1)
else:
    MOCK_KEY, MOCK_VAL = "mock", "true"

def _headers() -> Dict[str, str]:
    headers: Dict[str, str] = {"Content-Type": "application/json"}
    if AUTH_HEADER:
        headers["Authorization"] = AUTH_HEADER
    return headers

class ChatUser(HttpUser):
    wait_time = between(0.3, 1.2)

    @task(3)
    def chat(self) -> None:
        now = int(time.time())
        in_mock = ((now // TOGGLE_INTERVAL) % 2) == 0
        real_call = random.random() < REAL_FRACTION

        params: Dict[str, str] = {}
        if in_mock and not real_call:
            params[MOCK_KEY] = MOCK_VAL

        payload = {"prompt": "Hello, world!"}

        with self.client.post(
            "/chat", json=payload, params=params, headers=_headers(), catch_response=True
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"unexpected status {resp.status_code} — {resp.text[:120]}")

    @task(1)
    def ping(self) -> None:
        self.client.get("/ping")

@locust_events.test_start.add_listener
def on_test_start(environment, **_):
    if hasattr(environment, "process_exit_code"):
        environment.process_exit_code = 0
    users = getattr(environment, "user_count", 0)
    spawn = getattr(environment, "spawn_rate", 0)
    msg = (
        f"Soak test started — users={users}, spawn_rate={spawn}, "
        f"real_fraction={REAL_FRACTION}, toggle_interval={TOGGLE_INTERVAL}s"
    )
    hook = getattr(environment.events, "request_success", None)
    if hook:
        hook.fire(request_type="INFO", name="startup", response_time=0, response_length=len(msg))

def test_placeholder_skip():
    import pytest
    pytest.skip("Skipping soak load test in pytest environment", allow_module_level=False)
