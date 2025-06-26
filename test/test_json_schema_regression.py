# File: test/test_json_schema_regression.py
"""
Contract regression test for POST /chat using jsonschema + a RefResolver.
This makes $ref lookups against your full OpenAPI spec.
"""

import os
import json
from pathlib import Path

import httpx
import pytest
from jsonschema import validate, RefResolver, ValidationError

# Load the entire OpenAPI JSON document
SCHEMA_PATH = os.getenv(
    "OPENAPI_SCHEMA_PATH",
    os.path.join(os.path.dirname(__file__), "../openapi_schema.json"),
)
spec = json.loads(Path(SCHEMA_PATH).read_text())

# Drill down to the response schema fragment
resp_schema = spec["paths"]["/chat"]["post"]["responses"]["200"]["content"][
    "application/json"
]["schema"]

# Prepare a resolver capable of resolving "#/components/..." references
resolver = RefResolver.from_schema(spec)

BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
TOKEN = os.getenv("CHATBOT_TOKEN")  # as before

@pytest.fixture(scope="session")
def client():
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    return httpx.Client(base_url=BASE_URL, headers=headers, timeout=10.0)

def test_chat_contract(client):
    if not TOKEN:
        pytest.skip("CHATBOT_TOKEN not set; skipping authenticated contract test")

    payload = {"prompt": "ping"}
    r = client.post("/chat", json=payload)
    assert r.status_code == 200, f"Unexpected {r.status_code}: {r.text}"

    data = r.json()
    try:
        validate(instance=data, schema=resp_schema, resolver=resolver)
    except ValidationError as e:
        pytest.fail(f"JSON does not match schema: {e.message}")
