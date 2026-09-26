import pytest
from unittest.mock import patch, MagicMock

from liz.config.settings import Settings
from liz.infrastructure.ai.factory import create_provider
from liz.infrastructure.ai.base import AIProvider


def _settings(provider: str = "gemini", **kwargs) -> Settings:
    return Settings(
        ai_provider=provider,
        ai_model="test-model",
        gemini_api_key="test-key",
        openai_api_key="test-key",
        anthropic_api_key="test-key",
        deepseek_api_key="test-key",
        **kwargs,
    )


class TestFactory:
    def test_create_gemini_provider(self) -> None:
        with patch("liz.infrastructure.ai.gemini.GeminiProvider") as mock:
            mock.return_value = MagicMock(spec=AIProvider)
            provider = create_provider(_settings("gemini"))
            assert isinstance(provider, AIProvider)

    def test_create_openai_provider(self) -> None:
        with patch("liz.infrastructure.ai.openai.OpenAIProvider") as mock:
            mock.return_value = MagicMock(spec=AIProvider)
            provider = create_provider(_settings("openai"))
            assert isinstance(provider, AIProvider)

    def test_create_claude_provider(self) -> None:
        with patch("liz.infrastructure.ai.claude.ClaudeProvider") as mock:
            mock.return_value = MagicMock(spec=AIProvider)
            provider = create_provider(_settings("claude"))
            assert isinstance(provider, AIProvider)

    def test_create_deepseek_provider(self) -> None:
        with patch("liz.infrastructure.ai.deepseek.DeepSeekProvider") as mock:
            mock.return_value = MagicMock(spec=AIProvider)
            provider = create_provider(_settings("deepseek"))
            assert isinstance(provider, AIProvider)

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            create_provider(_settings("ollama"))

    def test_case_insensitive_provider(self) -> None:
        with patch("liz.infrastructure.ai.gemini.GeminiProvider") as mock:
            mock.return_value = MagicMock(spec=AIProvider)
            provider = create_provider(_settings("Gemini"))
            assert isinstance(provider, AIProvider)

    def test_factory_is_function(self) -> None:
        assert callable(create_provider)
