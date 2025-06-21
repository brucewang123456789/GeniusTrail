from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    XAI_API_KEY: str
    XAI_API_URL: str = "https://api.x.ai/v1/chat/completions"
    VELTRAX_MODEL: str = "grok-3-latest"
    VELTRAX_API_TOKEN: str | None = None
    CORS_ORIGINS: str = "*"
    PORT: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )


settings = Settings()
