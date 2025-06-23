"""
Project-wide settings object.

Key change: XAI_API_KEY is now *optional* (default ``None``), so the module
can be imported even when the key is not set – this prevents CI collection
errors.  Unknown environment variables are ignored.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ──────────────────────────────────────────────────────────────────────
    # External X-AI service (optional in CI)
    # ──────────────────────────────────────────────────────────────────────
    XAI_API_KEY: Optional[str] = None
    XAI_API_URL: str = "https://api.x.ai/v1/chat/completions"

    # ──────────────────────────────────────────────────────────────────────
    # Veltraxor runtime options
    # ──────────────────────────────────────────────────────────────────────
    VELTRAX_MODEL: str = "grok-3-latest"
    VELTRAX_API_TOKEN: Optional[str] = None
    CORS_ORIGINS: str = "*"
    PORT: int = 8000

    # pydantic-settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unknown env vars to stay flexible
    )


@lru_cache
def get_settings() -> Settings:
    """Return a singleton Settings instance."""
    return Settings()


# a ready-to-use global instance
settings = get_settings()
