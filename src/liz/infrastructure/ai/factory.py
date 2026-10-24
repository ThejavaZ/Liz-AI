from liz.config.settings import Settings
from liz.infrastructure.ai.base import AIProvider
from liz.infrastructure.ai.fallback import FallbackProvider
from liz.infrastructure.ai.models_registry import ResolvedModel, resolve_models


def _build_provider(
    provider_name: str,
    model: ResolvedModel,
    settings: Settings,
) -> AIProvider:
    if provider_name == "gemini":
        from liz.infrastructure.ai.gemini import GeminiProvider

        return GeminiProvider(api_key=settings.gemini_api_key, model=model.name)

    if provider_name == "openai":
        from liz.infrastructure.ai.openai import OpenAIProvider

        return OpenAIProvider(api_key=settings.openai_api_key, model=model.name)

    if provider_name == "claude":
        from liz.infrastructure.ai.claude import ClaudeProvider

        return ClaudeProvider(
            api_key=settings.anthropic_api_key, model=model.name
        )

    if provider_name == "deepseek":
        from liz.infrastructure.ai.deepseek import DeepSeekProvider

        return DeepSeekProvider(
            api_key=settings.deepseek_api_key, model=model.name
        )

    if provider_name == "ollama":
        from liz.infrastructure.ai.ollama import OllamaProvider

        return OllamaProvider(
            model=model.name, base_url=settings.ollama_base_url
        )

    raise ValueError(f"Unknown provider: {provider_name}")


def create_provider(settings: Settings) -> AIProvider:
    """Create the provider chain for the configured backend.

    The requested model is tried first; provider defaults follow as
    fallbacks so a stale or mistyped model name degrades gracefully.
    """
    provider_name = settings.ai_provider.lower()
    models = resolve_models(provider_name, settings.ai_model)

    providers = [
        _build_provider(provider_name, model, settings) for model in models
    ]

    if len(providers) == 1:
        return providers[0]

    return FallbackProvider(providers)
