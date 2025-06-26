"""Root-level pytest configuration.

This shim restores two private names that were removed from
jsonschema ≥ 4.22 so that downstream libraries (notably
Hypothesis-JSONSchema, and therefore Schemathesis) can import without
crashing.  No other behaviour is changed.
"""

from __future__ import annotations

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
        # In jsonschema ≤ 4.21 this still exists publicly.
        _jv._RefResolver = jsonschema.RefResolver  # type: ignore[attr-defined]
    except AttributeError:

        class _NoRefResolver:
            """Placeholder for the removed RefResolver.

            Any attempt to instantiate this will raise, which is acceptable
            because Hypothesis-JSONSchema only needs the name at import time.
            """

            def __init__(self, *_: object, **__: object) -> None:
                raise NotImplementedError(
                    "jsonschema.RefResolver has been removed; this is a stub."
                )

        _jv._RefResolver = _NoRefResolver  # type: ignore[assignment]
