from unittest.mock import patch

from liz.core.agent import Agent
from liz.core.models import Role
from liz.config.settings import Settings
from tests.conftest import MockAIProvider


def _make_settings(**kwargs) -> Settings:
    return Settings(
        ai_provider="gemini",
        ai_model="test-model",
        gemini_api_key="test-key",
        **kwargs,
    )


class TestAgent:
    def test_agent_creates_provider(self) -> None:
        with patch("liz.core.agent.create_provider") as mock_create:
            mock_create.return_value = MockAIProvider()
            agent = Agent(_make_settings(), "sys")
            mock_create.assert_called_once()

    def test_agent_chat_returns_response(self) -> None:
        with patch("liz.core.agent.create_provider") as mock_create:
            mock_create.return_value = MockAIProvider("Hello!")
            agent = Agent(_make_settings(), "sys")
            result = agent.chat("Hi")
            assert result == "Hello!"

    def test_agent_adds_to_context(self) -> None:
        with patch("liz.core.agent.create_provider") as mock_create:
            mock_create.return_value = MockAIProvider("reply")
            agent = Agent(_make_settings(), "sys")
            agent.chat("question")
            msgs = agent.context.get_messages()
            assert len(msgs) == 3
            assert msgs[1].role == Role.USER
            assert msgs[1].content == "question"
            assert msgs[2].role == Role.ASSISTANT
            assert msgs[2].content == "reply"

    def test_agent_passes_messages_to_provider(self) -> None:
        provider = MockAIProvider("ok")
        with patch("liz.core.agent.create_provider", return_value=provider):
            agent = Agent(_make_settings(), "sys")
            agent.chat("test")
            call_args = provider.chat_calls[0]
            messages = call_args["messages"]
            assert len(messages) == 2
            assert messages[0].role == Role.SYSTEM
            assert messages[1].role == Role.USER

    def test_agent_provider_error_propagates(self) -> None:
        with patch("liz.core.agent.create_provider") as mock_create:
            from tests.conftest import MockAIProvider as MP

            class FailingProvider(MP):
                def chat(self, messages, tools=None):
                    raise ConnectionError("API down")

            mock_create.return_value = FailingProvider()
            agent = Agent(_make_settings(), "sys")
            try:
                agent.chat("test")
                assert False, "Should have raised"
            except ConnectionError as e:
                assert "API down" in str(e)
