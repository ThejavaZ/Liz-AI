from liz.config.settings import Settings
from liz.infrastructure.ai.base import AIProvider


def create_provider(settings: Settings) -> AIProvider:
    provider_name = settings.ai_provider.lower()

    if provider_name == "gemini":
        from liz.infrastructure.ai.gemini import GeminiProvider

        return GeminiProvider(
            api_key=settings.gemini_api_key,
            model=settings.ai_model,
        )
    elif provider_name == "openai":
        from liz.infrastructure.ai.openai import OpenAIProvider

        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.ai_model,
        )
    elif provider_name == "claude":
        from liz.infrastructure.ai.claude import ClaudeProvider

        return ClaudeProvider(
            api_key=settings.anthropic_api_key,
            model=settings.ai_model,
        )
    elif provider_name == "deepseek":
        from liz.infrastructure.ai.deepseek import DeepSeekProvider

        return DeepSeekProvider(
            api_key=settings.deepseek_api_key,
            model=settings.ai_model,
        )
    else:
        raise ValueError(f"Unknown provider: {provider_name}")
