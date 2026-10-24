"""DeepSeek provider.

DeepSeek exposes an OpenAI-compatible API, so this subclasses the
OpenAI provider and only changes the base URL.
"""

from liz.infrastructure.ai.openai import OpenAIProvider


class DeepSeekProvider(OpenAIProvider):
    BASE_URL = "https://api.deepseek.com"

    def __init__(self, api_key: str | None, model: str) -> None:
        if not api_key:
            raise ValueError("DeepSeek API key is required")

        super().__init__(
            api_key=api_key,
            model=model,
            base_url=self.BASE_URL,
        )
