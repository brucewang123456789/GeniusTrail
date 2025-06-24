"""
Shared test-level fixtures for ALL unit-test suites.

Placing disable_auth here guarantees that token verification
is bypassed in every test file under `test/`.
"""

import pytest
from veltraxor import app, verify_token


@pytest.fixture(autouse=True, scope="session")
def disable_auth_for_unit_tests():
    """Disable Bearer-token enforcement globally during unit tests."""
    app.dependency_overrides[verify_token] = lambda: None
    yield
    app.dependency_overrides.pop(verify_token, None)

