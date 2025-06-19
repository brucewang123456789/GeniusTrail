from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

"""
Settings for Veltraxor (Grok-3 edition).
All values come from environment variables; no null bytes here.
"""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    XAI_API_KEY: str = os.getenv("XAI_API_KEY", "")
    XAI_API_URL: str = os.getenv(
        "XAI_API_URL",
        "https://api.grok.ai/v1/chat/completions",
    )
    VELTRAX_MODEL: str = os.getenv("VELTRAX_MODEL", "grok-3-latest")


settings = Settings()