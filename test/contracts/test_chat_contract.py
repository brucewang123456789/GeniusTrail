import os
import jsonschema
import pytest
from fastapi.testclient import TestClient
from api_server import app

client = TestClient(app)
TOKEN = os.getenv("VELTRAX_API_TOKEN", "")

schema = {
    "type": "object",
    "properties": {
        "response": {"type": "string"},
        "used_cot": {"type": "boolean"},
        "duration_ms": {"type": "integer"}
    },
    "required": ["response", "used_cot", "duration_ms"]
}

def test_chat_response_schema():
    if not TOKEN:
        pytest.skip("VELTRAX_API_TOKEN not set")
    headers = {"Authorization": f"Bearer {TOKEN}"}
    resp = client.post(
        "/chat",
        json={"prompt": "contract test", "history": []},
        headers=headers
    )
    assert resp.status_code == 200
    jsonschema.validate(resp.json(), schema)
