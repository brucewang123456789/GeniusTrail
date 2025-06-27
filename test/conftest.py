"""Test-level pytest configuration.

This file provides a dummy API token and also restores the same two
private jsonschema names so that Hypothesis-JSONSchema and Schemathesis
can import without crashing during test collection.
"""

import os
import jsonschema
import jsonschema.exceptions as _je
import jsonschema.validators as _jv

# ---------------------------------------------------------------------------
# Restore jsonschema.exceptions._RefResolutionError
# ---------------------------------------------------------------------------
if not hasattr(_je, "_RefResolutionError"):
    _je._RefResolutionError = _je.RefResolutionError  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Restore jsonschema.validators._RefResolver
# ---------------------------------------------------------------------------
if not hasattr(_jv, "_RefResolver"):
    try:
        _jv._RefResolver = jsonschema.RefResolver  # type: ignore[attr-defined]
    except AttributeError:

        class _NoRefResolver:
            """Stub for the removed RefResolver; only needed at import-time."""

            def __init__(self, *_: object, **__: object) -> None:
                raise NotImplementedError(
                    "jsonschema.RefResolver has been removed; this is a stub."
                )

        _jv._RefResolver = _NoRefResolver  # type: ignore[assignment]

# Provide a dummy API token so that any test depending on VELTRAX_API_TOKEN passes
os.environ.setdefault("VELTRAX_API_TOKEN", "dummy_token")
