# test/stress/test_smoke.py
#!/usr/bin/env python3
"""
Lightweight smoke-test script focused on /ping stability.

Before running concurrent /ping calls, performs a short retry loop to wait
until the service is up and responding. Only /ping is exercised in smoke mode.

Environment variables
---------------------
STRESS_BASE_URL       Base URL of the FastAPI instance (default: http://127.0.0.1:8000)
VELTRAX_API_TOKEN     (Not used for /ping, but required by script structure)
STRESS_CONCURRENCY    Parallel simulated users (default: 5)
STRESS_REQUESTS_PER_USER  Requests each user will issue (default: 10)
STRESS_TIMEOUT        Per-request timeout in seconds (default: 60.0)
PING_WAIT_RETRIES     How many times to retry /ping before giving up (default: 5)
PING_WAIT_INTERVAL    Seconds between retries (default: 1.0)
"""

from __future__ import annotations
import asyncio
import os
import sys
import time
from typing import Tuple

import httpx
from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
load_dotenv(override=False)

BASE_URL: str = os.getenv("STRESS_BASE_URL", "http://127.0.0.1:8000")
API_TOKEN: str | None = os.getenv("VELTRAX_API_TOKEN")
CONCURRENT_USERS: int = int(os.getenv("STRESS_CONCURRENCY", "5"))
REQUESTS_PER_USER: int = int(os.getenv("STRESS_REQUESTS_PER_USER", "10"))
TIMEOUT_SECONDS: float = float(os.getenv("STRESS_TIMEOUT", "60.0"))

# Pre-check settings
PING_WAIT_RETRIES: int = int(os.getenv("PING_WAIT_RETRIES", "5"))
PING_WAIT_INTERVAL: float = float(os.getenv("PING_WAIT_INTERVAL", "1.0"))

# --------------------------------------------------------------------------- #
# Request helpers
# --------------------------------------------------------------------------- #
async def ping(client: httpx.AsyncClient) -> None:
    """Simple liveness probe."""
    r = await client.get(f"{BASE_URL}/ping", timeout=TIMEOUT_SECONDS)
    r.raise_for_status()


# --------------------------------------------------------------------------- #
# Pre-check: wait until /ping responds 200
# --------------------------------------------------------------------------- #
async def wait_for_ping() -> None:
    last_exc: Exception | None = None
    async with httpx.AsyncClient() as client:
        for attempt in range(1, PING_WAIT_RETRIES + 1):
            try:
                r = await client.get(f"{BASE_URL}/ping", timeout=TIMEOUT_SECONDS)
                r.raise_for_status()
                print(f"/ping ok on attempt {attempt}")
                return
            except Exception as exc:
                last_exc = exc
                print(f"/ping attempt {attempt} failed: {exc!r}, retrying in {PING_WAIT_INTERVAL}s")
                await asyncio.sleep(PING_WAIT_INTERVAL)
    print(f"Error: /ping did not succeed after {PING_WAIT_RETRIES} attempts")
    if last_exc:
        print(f"Last error: {last_exc!r}")
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Simulation (only ping)
# --------------------------------------------------------------------------- #
async def user_simulation(user_id: int) -> Tuple[int, int]:
    """
    Simulate one virtual user issuing a small burst of /ping requests.
    Returns (success_count, failure_count)
    """
    successes, failures = 0, 0
    async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
        for _ in range(REQUESTS_PER_USER):
            try:
                await ping(client)
                successes += 1
            except Exception as exc:
                failures += 1
                print(f"[user {user_id}] ping failed: {exc!r}")
    return successes, failures


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
async def main() -> None:
    # API_TOKEN presence check is kept to align with structure, though not used here
    if not API_TOKEN:
        sys.exit("Error: missing VELTRAX_API_TOKEN environment variable")

    # Pre-check service readiness
    await wait_for_ping()

    # Run concurrent /ping simulations
    tasks = [asyncio.create_task(user_simulation(uid)) for uid in range(CONCURRENT_USERS)]
    results = await asyncio.gather(*tasks)

    total_success = sum(s for s, _ in results)
    total_fail = sum(f for _, f in results)

    print("\n=== Stress-smoke summary ===")
    print(f"Base URL          : {BASE_URL}")
    print(f"Virtual users     : {CONCURRENT_USERS}")
    print(f"Requests per user : {REQUESTS_PER_USER}")
    print(f"Successful pings  : {total_success}")
    print(f"Failed pings      : {total_fail}")

    sys.exit(0 if total_fail == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
