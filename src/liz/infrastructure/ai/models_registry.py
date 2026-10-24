"""Central registry of models per provider.

Keeping model knowledge in one place means adding or updating a model
never requires touching Settings or the Agent.
"""

from dataclasses import dataclass

PROVIDERS = ("gemini", "openai", "claude", "deepseek", "ollama")

DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
    "claude": "claude-sonnet-4-20250514",
    "deepseek": "deepseek-chat",
    "ollama": "llama3.1",
}

FALLBACK_MODELS: dict[str, list[str]] = {
    "gemini": ["gemini-1.5-flash", "gemini-1.5-pro"],
    "openai": ["gpt-4o", "gpt-4o-mini"],
    "claude": ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"],
    "deepseek": ["deepseek-chat", "deepseek-reasoner"],
    "ollama": ["llama3.1", "qwen2.5", "mistral"],
}

KNOWN_MODELS: dict[str, list[str]] = {
    "gemini": [
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ],
    "openai": [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4.1",
        "gpt-4.1-mini",
        "o4-mini",
    ],
    "claude": [
        "claude-sonnet-4-20250514",
        "claude-opus-4-20250514",
        "claude-3-5-sonnet-latest",
        "claude-3-5-haiku-latest",
    ],
    "deepseek": [
        "deepseek-chat",
        "deepseek-reasoner",
    ],
    "ollama": [
        "llama3.1",
        "qwen2.5",
        "mistral",
        "gemma2",
        "phi3",
    ],
}


@dataclass(frozen=True)
class ResolvedModel:
    """A model to try, and whether it is the last candidate in the chain."""

    name: str
    is_fallback: bool = False


def resolve_models(provider: str, requested: str | None) -> list[ResolvedModel]:
    """Return the ordered list of models to try for a provider.

    If the user asked for a specific model it is tried first, then the
    provider defaults act as fallbacks. If no model was requested the
    default chain is used directly.
    """
    provider = provider.lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(
            f"Unknown provider: {provider}. Supported: {', '.join(PROVIDERS)}"
        )

    chain: list[str] = []
    if requested:
        chain.append(requested)

    for candidate in [DEFAULT_MODELS[provider], *FALLBACK_MODELS[provider]]:
        if candidate not in chain:
            chain.append(candidate)

    return [
        ResolvedModel(name=name, is_fallback=index > 0)
        for index, name in enumerate(chain)
    ]


def list_models(provider: str) -> list[str]:
    """Return known model names for a provider."""
    provider = provider.lower()
    return list(KNOWN_MODELS.get(provider, []))


def default_model(provider: str) -> str:
    """Return the default model for a provider."""
    provider = provider.lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(
            f"Unknown provider: {provider}. Supported: {', '.join(PROVIDERS)}"
        )
    return DEFAULT_MODELS[provider]
