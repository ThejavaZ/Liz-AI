from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    ai_provider: str = "gemini"
    ai_model: str | None = None

    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    deepseek_api_key: str | None = None
    anthropic_api_key: str | None = None

    ollama_base_url: str = "http://localhost:11434/v1"

    data_dir: str = "data"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
