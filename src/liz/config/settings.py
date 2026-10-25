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

    # voice (Phase 7)
    voice_language: str = "es"
    voice_record_seconds: int = 5
    voice_wake_words: str = "hola liz, hey liz, oye liz"
    whisper_bin: str | None = None
    whisper_model: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
