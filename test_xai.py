"""
Integration test for an external X-AI endpoint.

If the required credentials (XAI_API_KEY / XAI_API_URL) are **not** present,
the whole module is skipped – this keeps CI green without exposing secrets.
"""

from __future__ import annotations

import os
from typing import Any  # ← added
import pytest
import httpx
from dotenv import load_dotenv

load_dotenv()  # load variables from .env if available

# Read raw values (may be None)
API_URL_RAW: str | None = os.getenv("XAI_API_URL")
API_KEY_RAW: str | None = os.getenv("XAI_API_KEY")

# Skip entire module if missing
if not API_URL_RAW or not API_KEY_RAW:
    pytest.skip(
        "XAI credentials missing – skipping external XAI integration tests",
        allow_module_level=True,
    )

# At this point, both are non-None, so bind to pure str
API_URL: str = API_URL_RAW
API_KEY: str = API_KEY_RAW

HEADERS: dict[str, str] = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

PAYLOAD: dict[str, Any] = {
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
