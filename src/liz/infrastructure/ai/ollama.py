"""Local model support via Ollama.

Ollama exposes an OpenAI-compatible API on localhost:11434, so this
provider reuses the OpenAI client with a custom base URL. No API key is
required, which makes it the zero-dependency option for offline use.
"""

from liz.infrastructure.ai.openai import OpenAIProvider

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"


class OllamaProvider(OpenAIProvider):
    def __init__(self, model: str, base_url: str = OLLAMA_BASE_URL) -> None:
        super().__init__(
            api_key=OLLAMA_API_KEY,
            model=model,
            base_url=base_url,
        )
