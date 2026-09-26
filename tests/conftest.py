from unittest.mock import MagicMock
from typing import Any

import pytest

from liz.infrastructure.ai.base import AIProvider
from liz.core.models import AIResponse, Message


class MockAIProvider(AIProvider):
    def __init__(self, response_text: str = "Hello!") -> None:
        self.response_text = response_text
        self.chat_calls: list[dict] = []

    def chat(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> AIResponse:
        self.chat_calls.append({"messages": messages, "tools": tools})
        return AIResponse(
            content=self.response_text,
            model="mock-model",
            usage={"input_tokens": 10, "output_tokens": 5},
        )


@pytest.fixture
def mock_provider() -> MockAIProvider:
    return MockAIProvider()


@pytest.fixture
def mock_provider_with_error() -> MagicMock:
    provider = MagicMock(spec=AIProvider)
    provider.chat.side_effect = ConnectionError("API unavailable")
    return provider
