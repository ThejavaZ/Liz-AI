from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    ai_provider: str = "gemini"
    ai_model: str = "gemini-2.0-flash"

    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    deepseek_api_key: str | None = None
    anthropic_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )