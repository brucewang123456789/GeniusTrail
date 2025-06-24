from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # External X-AI service (optional in CI)
    XAI_API_KEY: Optional[str] = None
    XAI_API_URL: str = "https://api.x.ai/v1/chat/completions"

    # Veltraxor runtime options
    VELTRAX_MODEL: str = "grok-3-latest"
    VELTRAX_API_TOKEN: Optional[str] = None
    CORS_ORIGINS: str = "*"
    PORT: int = 8000

    # Stress test & dynamic CoT tuning
    CHAT_TIMEOUT: int = 60      # Client-side timeout for /chat (seconds)
    COT_MAX_ROUNDS: int = 3     # Default max CoT iterations; override for smoke tests
    SMOKE_TEST: bool = False    # If True, smoke test mode disables chat and CoT integration
    STRESS_TIMEOUT: float = 60.0  # Default per-request timeout for stress-test script

    # pydantic-settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unknown env vars
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


# Ready-to-use global instance
settings = get_settings()