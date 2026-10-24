"""Fallback wrapper that tries several model-backed providers in order.

Each candidate is usually the same provider class configured with a
different model, so a missing model or an exhausted quota degrades into
the next candidate instead of failing the whole request.
"""

from typing import Any

from liz.core.models import AIResponse, Message
from liz.infrastructure.ai.base import AIProvider

RETRYABLE_MARKERS = (
    "429",
    "503",
    "500",
    "resource_exhausted",
    "resource exhausted",
    "rate limit",
    "ratelimit",
    "overloaded",
    "unavailable",
    "not found",
    "404",
    "invalid model",
    "model_not_found",
    "does not exist",
    "quota",
)


def is_retryable(error: Exception) -> bool:
    """Return True when the error is likely transient or model-specific."""
    message = str(error).lower()
    return any(marker in message for marker in RETRYABLE_MARKERS)


class FallbackProvider(AIProvider):
    def __init__(self, providers: list[AIProvider]) -> None:
        if not providers:
            raise ValueError("FallbackProvider requires at least one provider")
        self.providers = providers

    @property
    def active(self) -> AIProvider:
        return self.providers[0]

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> AIResponse:
        last_error: Exception | None = None

        for index, provider in enumerate(self.providers):
            is_last = index == len(self.providers) - 1
            try:
                return provider.chat(messages, tools)
            except Exception as error:
                if is_last or not is_retryable(error):
                    raise
                last_error = error

        raise last_error if last_error else RuntimeError("No providers available")
