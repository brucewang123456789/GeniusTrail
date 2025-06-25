# config.py

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # External X-AI service
    XAI_API_KEY: Optional[str] = None
    XAI_API_URL: str = "https://api.x.ai/v1/chat/completions"

    # Veltraxor runtime options
    VELTRAX_MODEL: str = "grok-3-latest"
    VELTRAX_API_TOKEN: Optional[str] = None
    VELTRAX_SYSTEM_PROMPT: str = ""
    CORS_ORIGINS: str = "*"
    PORT: int = 8000

    # Chat & CoT tuning
    CHAT_TIMEOUT: int = 60
    COT_MAX_ROUNDS: int = 3
    SMOKE_TEST: bool = False

    # Stress & deep stress configuration
    STRESS_TIMEOUT: float = 60.0
    DEEP_STRESS: bool = False
    DEEP_CONCURRENCY: int = 50
    DEEP_REQUESTS_PER_CLIENT: int = 100
    LATENCY_THRESHOLDS: str = "500,1000,2000"

    # Cost-saving and mock flags
    COST_SAVING_TEST: bool = False    # Quick 3×ping/chat mode
    MOCK_LLM: bool = False            # Local stub mode for LLMClient

    # Retry configuration
    MAX_RETRIES: int = 2
    BACKOFF_FACTOR: float = 0.5

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
