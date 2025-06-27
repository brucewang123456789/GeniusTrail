# test/deep_stress/deep_stress_test.py

"""
Deep stress test script for comprehensive load and resilience verification.

Behavior:
- If COST_SAVING_TEST=1: perform 3 pings and 3 chat calls, then exit.
- Else if DEEP_STRESS=1: perform full deep stress test.
- Otherwise: exit without running.
"""

import os
import sys
import asyncio
import time
import statistics

import httpx

# --------------------------------------------------------------------------- #
# Quick cost-saving test
# --------------------------------------------------------------------------- #
if os.getenv("COST_SAVING_TEST", "0") == "1":
    BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
    API_TOKEN = os.getenv("API_TOKEN")
    HEADERS = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"} if API_TOKEN else {}

    async def quick_check():
        async with httpx.AsyncClient(timeout=httpx.Timeout(60)) as client:
            for i in range(3):
                r = await client.get(f"{BASE_URL}/ping")
                print(f"[quick] ping {i}: {r.status_code}")
                r.raise_for_status()
            payload = {"prompt": "Quick cost-saving test"}
            for i in range(3):
                r = await client.post(f"{BASE_URL}/chat", json=payload, headers=HEADERS)
                print(f"[quick] chat {i}: {r.status_code}")
                r.raise_for_status()
        print("Quick cost-saving test passed")
        sys.exit(0)

    asyncio.run(quick_check())

# --------------------------------------------------------------------------- #
# Guard: only run in deep stress mode
# --------------------------------------------------------------------------- #
if os.getenv("DEEP_STRESS", "0") != "1":
    print("[deep_stress] DEEP_STRESS not enabled; skipping deep stress tests.")
    sys.exit(0)

# --------------------------------------------------------------------------- #
# Configuration for deep stress
# --------------------------------------------------------------------------- #
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
API_TOKEN = os.getenv("API_TOKEN")
DEEP_CONCURRENCY = int(os.getenv("DEEP_CONCURRENCY", "50"))
DEEP_REQUESTS_PER_CLIENT = int(os.getenv("DEEP_REQUESTS_PER_CLIENT", "100"))
STRESS_TIMEOUT = int(os.getenv("STRESS_TIMEOUT", "120"))

THRESHOLDS = list(map(int, os.getenv("LATENCY_THRESHOLDS", "500,1000,2000").split(",")))
HEADERS = {"Authorization": f"Bearer {API_TOKEN}", "Content-Type": "application/json"} if API_TOKEN else {}

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
async def ping(client: httpx.AsyncClient) -> float:
    start = time.perf_counter()
    r = await client.get(f"{BASE_URL}/ping")
    r.raise_for_status()
    return (time.perf_counter() - start) * 1000

async def chat(client: httpx.AsyncClient, payload: dict) -> float:
    start = time.perf_counter()
    r = await client.post(f"{BASE_URL}/chat", json=payload, headers=HEADERS)
    r.raise_for_status()
    return (time.perf_counter() - start) * 1000

async def client_task(client_id: int, errors: list[Exception]) -> list[float]:
    latencies: list[float] = []
    timeout = httpx.Timeout(STRESS_TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            latencies.append(await ping(client))
        except Exception as e:
            print(f"[c{client_id}] ping failed: {e}")
            errors.append(e)
        for i in range(DEEP_REQUESTS_PER_CLIENT):
            payload = {"prompt": "Stress test payload " + "x" * 1000}
            try:
                latencies.append(await chat(client, payload))
            except Exception as e:
                print(f"[c{client_id}] chat {i} failed: {e}")
                errors.append(e)
    return latencies

# --------------------------------------------------------------------------- #
# Main deep stress
# --------------------------------------------------------------------------- #
async def main() -> None:
    if not API_TOKEN:
        print("API_TOKEN not set; aborting.")
        sys.exit(1)

    errors: list[Exception] = []
    tasks = [asyncio.create_task(client_task(i, errors)) for i in range(DEEP_CONCURRENCY)]
    all_latencies: list[float] = []
    results = await asyncio.gather(*tasks)

    # 将每个任务返回的延迟列表依次合并
    for latency_list in results:
        all_latencies.extend(latency_list)

    if errors:
        print("Deep stress test encountered errors.")
        sys.exit(1)

    p50 = statistics.quantiles(all_latencies, n=100)[49]
    p95 = statistics.quantiles(all_latencies, n=100)[94]
    p99 = statistics.quantiles(all_latencies, n=100)[98]

    print(f"Requests total : {len(all_latencies)}")
    print(f"p50 latency    : {p50:.1f} ms (threshold {THRESHOLDS[0]})")
    print(f"p95 latency    : {p95:.1f} ms (threshold {THRESHOLDS[1]})")
    print(f"p99 latency    : {p99:.1f} ms (threshold {THRESHOLDS[2]})")

    if p50 > THRESHOLDS[0] or p95 > THRESHOLDS[1] or p99 > THRESHOLDS[2]:
        print("Latency thresholds exceeded; deep stress test failed.")
        sys.exit(1)

    print("Deep stress test passed within latency thresholds.")
    sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())
