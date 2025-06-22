#!/usr/bin/env python3
"""
check_env.py

Load environment variables from .env and print key values to verify they are set.
Usage: from the project root (where .env resides), run `python check_env.py`.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

# Try importing project settings if available
settings: Any | None  # single declaration avoids redefinition
try:
    from config import settings as _settings  # config.py should define `settings`

    settings = _settings
except ImportError:
    settings = None


def print_env_var(name: str, mask: bool = False) -> None:
    """
    Print the value of an environment variable.
    If mask=True, show only the first and last few characters to avoid full exposure.
    """
    val = os.getenv(name)
    if not val:
        print(f"{name}: <not set or empty>")
    elif mask and len(val) > 8:
        # Show first 4 and last 4 characters
        print(f"{name}: {val[:4]}...{val[-4:]} (masked)")
    else:
        print(f"{name}: {val}")


def main() -> None:
    # Load .env from current directory
    load_dotenv()

    print("=== Environment Variables Check ===")
    keys = [
        "XAI_API_URL",
        "XAI_API_KEY",
        "VELTRAX_MODEL",
        "VELTRAX_API_TOKEN",
        "CORS_ORIGINS",
    ]
    for key in keys:
        print_env_var(key, mask=key.endswith(("KEY", "TOKEN")))

    if settings is not None:
        print("\n=== config.Settings Values ===")
        for attr in [
            "XAI_API_URL",
            "XAI_API_KEY",
            "VELTRAX_MODEL",
            "VELTRAX_API_URL",
            "VELTRAX_API_KEY",
        ]:
            val = getattr(settings, attr, None)
            if not val:
                print(f"settings.{attr}: <not set or empty>")
            elif "KEY" in attr or "TOKEN" in attr:
                if isinstance(val, str) and len(val) > 8:
                    print(f"settings.{attr}: {val[:4]}...{val[-4:]} (masked)")
                else:
                    print(f"settings.{attr}: {val}")
            else:
                print(f"settings.{attr}: {val}")
    else:
        print(
            "\nconfig.Settings not imported: please check config.py and confirm it "
            "defines a 'settings' instance."
        )

    cwd = os.getcwd()
    print(f"\nCurrent working directory: {cwd}")
    print("Make sure you run this script from the project root where .env is located.")


if __name__ == "__main__":
    main()
