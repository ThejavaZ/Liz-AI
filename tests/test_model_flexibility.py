import pytest
from unittest.mock import MagicMock, patch

from liz.core.models import AIResponse, Message, Role
from liz.infrastructure.ai.base import AIProvider
from liz.infrastructure.ai.fallback import FallbackProvider, is_retryable
from liz.infrastructure.ai.models_registry import (
    DEFAULT_MODELS,
    PROVIDERS,
    ResolvedModel,
    default_model,
    list_models,
    resolve_models,
)
from liz.config.settings import Settings
from liz.infrastructure.ai.factory import create_provider
from liz.cli.commands import CommandHandler
from liz.core.context import Context
from liz.tools.registry import ToolRegistry


class _FakeProvider(AIProvider):
    def __init__(self, model: str, response=None, error=None) -> None:
        self.model = model
        self.response = response
        self.error = error
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        if self.error:
            raise self.error
        return self.response or AIResponse(content=f"ok from {self.model}", model=self.model)


class TestResolveModels:
    def test_requested_model_first(self) -> None:
        chain = resolve_models("gemini", "my-custom-model")
        assert chain[0].name == "my-custom-model"
        assert chain[0].is_fallback is False

    def test_default_used_when_none_requested(self) -> None:
        chain = resolve_models("gemini", None)
        assert chain[0].name == DEFAULT_MODELS["gemini"]

    def test_fallbacks_marked(self) -> None:
        chain = resolve_models("gemini", "custom")
        assert chain[0].is_fallback is False
        assert all(item.is_fallback for item in chain[1:])

    def test_no_duplicate_models(self) -> None:
        chain = resolve_models("openai", "gpt-4o")
        names = [item.name for item in chain]
        assert len(names) == len(set(names))

    def test_default_chain_when_requesting_default(self) -> None:
        chain = resolve_models("gemini", DEFAULT_MODELS["gemini"])
        names = [item.name for item in chain]
        assert names[0] == DEFAULT_MODELS["gemini"]
        assert len(names) > 1

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown provider"):
            resolve_models("bogus", None)

    def test_all_providers_have_defaults(self) -> None:
        for provider in PROVIDERS:
            assert provider in DEFAULT_MODELS


class TestListModels:
    def test_returns_known_models(self) -> None:
        models = list_models("gemini")
        assert "gemini-2.0-flash" in models

    def test_unknown_provider_empty(self) -> None:
        assert list_models("bogus") == []

    def test_case_insensitive(self) -> None:
        assert list_models("Gemini") == list_models("gemini")


class TestDefaultModel:
    def test_returns_default(self) -> None:
        assert default_model("openai") == DEFAULT_MODELS["openai"]

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            default_model("bogus")


class TestIsRetryable:
    def test_rate_limit(self) -> None:
        assert is_retryable(Exception("429 RESOURCE_EXHAUSTED"))

    def test_unavailable(self) -> None:
        assert is_retryable(Exception("503 UNAVAILABLE"))

    def test_model_not_found(self) -> None:
        assert is_retryable(Exception("404 model not found"))

    def test_quota(self) -> None:
        assert is_retryable(Exception("quota exceeded"))

    def test_non_retryable(self) -> None:
        assert not is_retryable(Exception("invalid api key"))

    def test_auth_error_not_retryable(self) -> None:
        assert not is_retryable(Exception("401 unauthorized"))


class TestFallbackProvider:
    def test_returns_first_success(self) -> None:
        first = _FakeProvider("model-a")
        second = _FakeProvider("model-b")
        provider = FallbackProvider([first, second])

        result = provider.chat([Message(role=Role.USER, content="hi")])

        assert "model-a" in result.content
        assert second.calls == 0

    def test_falls_back_on_retryable_error(self) -> None:
        first = _FakeProvider("model-a", error=Exception("429 rate limit"))
        second = _FakeProvider("model-b")
        provider = FallbackProvider([first, second])

        result = provider.chat([Message(role=Role.USER, content="hi")])

        assert "model-b" in result.content

    def test_no_fallback_on_auth_error(self) -> None:
        first = _FakeProvider("model-a", error=Exception("401 unauthorized"))
        second = _FakeProvider("model-b")
        provider = FallbackProvider([first, second])

        with pytest.raises(Exception, match="401"):
            provider.chat([Message(role=Role.USER, content="hi")])
        assert second.calls == 0

    def test_raises_last_error_when_exhausted(self) -> None:
        first = _FakeProvider("model-a", error=Exception("429 rate limit"))
        second = _FakeProvider("model-b", error=Exception("503 unavailable"))
        provider = FallbackProvider([first, second])

        with pytest.raises(Exception, match="503"):
            provider.chat([Message(role=Role.USER, content="hi")])

    def test_single_provider_passthrough(self) -> None:
        only = _FakeProvider("solo")
        provider = FallbackProvider([only])

        result = provider.chat([Message(role=Role.USER, content="hi")])

        assert "solo" in result.content

    def test_empty_providers_raises(self) -> None:
        with pytest.raises(ValueError):
            FallbackProvider([])

    def test_active_property(self) -> None:
        first = _FakeProvider("model-a")
        second = _FakeProvider("model-b")
        provider = FallbackProvider([first, second])
        assert provider.active is first


class TestFactoryModelChain:
    def _settings(self, provider: str, model=None) -> Settings:
        return Settings(
            ai_provider=provider,
            ai_model=model,
            gemini_api_key="k",
            openai_api_key="k",
            anthropic_api_key="k",
            deepseek_api_key="k",
        )

    def test_factory_returns_fallback_provider(self) -> None:
        with patch("liz.infrastructure.ai.gemini.GeminiProvider") as mock:
            mock.side_effect = lambda **kw: _FakeProvider(kw["model"])
            provider = create_provider(self._settings("gemini"))
            assert isinstance(provider, FallbackProvider)
            assert len(provider.providers) > 1

    def test_factory_chain_starts_with_requested_model(self) -> None:
        with patch("liz.infrastructure.ai.gemini.GeminiProvider") as mock:
            mock.side_effect = lambda **kw: _FakeProvider(kw["model"])
            provider = create_provider(
                self._settings("gemini", "custom-model")
            )
            assert provider.providers[0].model == "custom-model"

    def test_ollama_factory_no_api_key_needed(self) -> None:
        settings = Settings(ai_provider="ollama", ai_model="llama3.1")
        provider = create_provider(settings)
        assert isinstance(provider, FallbackProvider)
        assert provider.active.model == "llama3.1"

    def test_ollama_base_url_configurable(self) -> None:
        settings = Settings(
            ai_provider="ollama",
            ai_model="llama3.1",
            ollama_base_url="http://other:11434/v1",
        )
        provider = create_provider(settings)
        assert "other:11434" in str(provider.active.client.base_url)


class TestModelsCommand:
    def _handler(self, provider="gemini", model=None) -> CommandHandler:
        settings = Settings(
            ai_provider=provider,
            ai_model=model,
            gemini_api_key="k",
        )
        return CommandHandler(settings, Context("sys"), ToolRegistry())

    def test_lists_models(self) -> None:
        result = self._handler().handle("/models")
        assert "gemini" in result.output
        assert "gemini-2.0-flash" in result.output

    def test_marks_default(self) -> None:
        result = self._handler().handle("/models")
        assert "default" in result.output

    def test_marks_requested(self) -> None:
        result = self._handler(model="gemini-1.5-pro").handle("/models")
        assert "requested" in result.output

    def test_no_request_note(self) -> None:
        result = self._handler(model=None).handle("/models")
        assert "No model requested" in result.output

    def test_unknown_provider(self) -> None:
        result = self._handler(provider="bogus").handle("/models")
        assert "No model list available" in result.output
