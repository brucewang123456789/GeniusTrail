"""
Integration test for an external X-AI endpoint.

If the required credentials (XAI_API_KEY / XAI_API_URL) are **not** present,
the whole module is skipped – this keeps CI green without exposing secrets.
"""

from __future__ import annotations

import os
import pytest
import httpx
from dotenv import load_dotenv

load_dotenv()  # load variables from .env if available

API_URL = os.getenv("XAI_API_URL")
API_KEY = os.getenv("XAI_API_KEY")

# ─────────────────────────────────────────────────────────────────────────
# Automatically skip tests when credentials are missing
# ─────────────────────────────────────────────────────────────────────────
if not API_URL or not API_KEY:
    pytest.skip(
        "XAI credentials missing – skipping external XAI integration tests",
        allow_module_level=True,
    )

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

PAYLOAD = {
    "model": "grok-3-latest",
    "messages": [
        {"role": "system", "content": "You are a test assistant."},
        {
            "role": "user",
            "content": "Testing. Just say hi and hello world and nothing else.",
        },
    ],
    "stream": False,
}


def test_xai_endpoint_healthcheck() -> None:
    """Basic liveness check against the X-AI chat endpoint."""
    resp = httpx.post(API_URL, headers=HEADERS, json=PAYLOAD, timeout=10.0)
    assert resp.status_code == 200, f"Unexpected status {resp.status_code}"
    assert "hello" in resp.text.lower()
