import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from liz.core.models import AIResponse, Message, Role
from liz.infrastructure.ai.base import AIProvider
from liz.infrastructure.ai.gemini import GeminiProvider
from liz.infrastructure.ai.openai import OpenAIProvider
from liz.infrastructure.ai.claude import ClaudeProvider
from liz.infrastructure.ai.deepseek import DeepSeekProvider


class TestGeminiProvider:
    def test_requires_api_key(self) -> None:
        with pytest.raises(ValueError, match="Gemini API key is required"):
            GeminiProvider(api_key=None, model="test-model")

    def test_requires_api_key_empty(self) -> None:
        with pytest.raises(ValueError, match="Gemini API key is required"):
            GeminiProvider(api_key="", model="test-model")

    @patch("liz.infrastructure.ai.gemini.genai.Client")
    def test_creates_client(self, mock_client_class: MagicMock) -> None:
        provider = GeminiProvider(api_key="test-key", model="test-model")
        mock_client_class.assert_called_once_with(api_key="test-key")
        assert provider.model == "test-model"

    @patch("liz.infrastructure.ai.gemini.genai.Client")
    def test_chat_returns_response(self, mock_client_class: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_chat = MagicMock()
        mock_client.chats.create.return_value = mock_chat

        mock_part = MagicMock()
        mock_part.text = "Hello from Gemini"
        mock_part.function_call = None

        mock_content = MagicMock()
        mock_content.parts = [mock_part]

        mock_candidate = MagicMock()
        mock_candidate.content = mock_content

        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 5
        mock_chat.send_message.return_value = mock_response

        provider = GeminiProvider(api_key="test-key", model="test-model")
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful"),
            Message(role=Role.USER, content="Hi"),
        ]

        result = provider.chat(messages)

        assert isinstance(result, AIResponse)
        assert result.content == "Hello from Gemini"
        assert result.model == "test-model"
        assert result.usage["input_tokens"] == 10
        assert result.usage["output_tokens"] == 5

    @patch("liz.infrastructure.ai.gemini.genai.Client")
    def test_chat_formats_messages(self, mock_client_class: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_chat = MagicMock()
        mock_client.chats.create.return_value = mock_chat

        mock_response = MagicMock()
        mock_response.text = "Response"
        mock_response.usage_metadata.prompt_token_count = 5
        mock_response.usage_metadata.candidates_token_count = 3
        mock_chat.send_message.return_value = mock_response

        provider = GeminiProvider(api_key="test-key", model="test-model")
        messages = [
            Message(role=Role.SYSTEM, content="System prompt"),
            Message(role=Role.USER, content="User message"),
            Message(role=Role.ASSISTANT, content="Assistant message"),
            Message(role=Role.USER, content="Follow up"),
        ]

        provider.chat(messages)

        mock_client.chats.create.assert_called_once()
        mock_chat.send_message.assert_called_once()


class TestOpenAIProvider:
    def test_requires_api_key(self) -> None:
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            OpenAIProvider(api_key=None, model="test-model")

    def test_requires_api_key_empty(self) -> None:
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            OpenAIProvider(api_key="", model="test-model")

    @patch("liz.infrastructure.ai.openai.OpenAI")
    def test_creates_client(self, mock_client_class: MagicMock) -> None:
        provider = OpenAIProvider(api_key="test-key", model="gpt-4")
        mock_client_class.assert_called_once_with(api_key="test-key")
        assert provider.model == "gpt-4"

    @patch("liz.infrastructure.ai.openai.OpenAI")
    def test_chat_returns_response(self, mock_client_class: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello from OpenAI"))]
        mock_response.model = "gpt-4"
        mock_response.usage.prompt_token_count = 10
        mock_response.usage.completion_token_count = 5
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(api_key="test-key", model="gpt-4")
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful"),
            Message(role=Role.USER, content="Hi"),
        ]

        result = provider.chat(messages)

        assert isinstance(result, AIResponse)
        assert result.content == "Hello from OpenAI"
        assert result.model == "gpt-4"
        assert result.usage["input_tokens"] == 10
        assert result.usage["output_tokens"] == 5

    @patch("liz.infrastructure.ai.openai.OpenAI")
    def test_chat_formats_messages(self, mock_client_class: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_response.model = "gpt-4"
        mock_response.usage = None
        mock_client.chat.completions.create.return_value = mock_response

        provider = OpenAIProvider(api_key="test-key", model="gpt-4")
        messages = [
            Message(role=Role.SYSTEM, content="System"),
            Message(role=Role.USER, content="User"),
            Message(role=Role.ASSISTANT, content="Assistant"),
        ]

        provider.chat(messages)

        call_args = mock_client.chat.completions.create.call_args
        oai_messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        assert len(oai_messages) == 3
        assert oai_messages[0]["role"] == "system"
        assert oai_messages[1]["role"] == "user"
        assert oai_messages[2]["role"] == "assistant"


class TestClaudeProvider:
    def test_requires_api_key(self) -> None:
        with pytest.raises(ValueError, match="Anthropic API key is required"):
            ClaudeProvider(api_key=None, model="test-model")

    def test_requires_api_key_empty(self) -> None:
        with pytest.raises(ValueError, match="Anthropic API key is required"):
            ClaudeProvider(api_key="", model="test-model")

    @patch("liz.infrastructure.ai.claude.anthropic.Anthropic")
    def test_creates_client(self, mock_client_class: MagicMock) -> None:
        provider = ClaudeProvider(api_key="test-key", model="claude-3")
        mock_client_class.assert_called_once_with(api_key="test-key")
        assert provider.model == "claude-3"

    @patch("liz.infrastructure.ai.claude.anthropic.Anthropic")
    def test_chat_returns_response(self, mock_client_class: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Hello from Claude")]
        mock_response.model = "claude-3"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 5
        mock_client.messages.create.return_value = mock_response

        provider = ClaudeProvider(api_key="test-key", model="claude-3")
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful"),
            Message(role=Role.USER, content="Hi"),
        ]

        result = provider.chat(messages)

        assert isinstance(result, AIResponse)
        assert result.content == "Hello from Claude"
        assert result.model == "claude-3"
        assert result.usage["input_tokens"] == 10
        assert result.usage["output_tokens"] == 5

    @patch("liz.infrastructure.ai.claude.anthropic.Anthropic")
    def test_chat_formats_messages(self, mock_client_class: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Response")]
        mock_response.model = "claude-3"
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 3
        mock_client.messages.create.return_value = mock_response

        provider = ClaudeProvider(api_key="test-key", model="claude-3")
        messages = [
            Message(role=Role.SYSTEM, content="System"),
            Message(role=Role.USER, content="User"),
            Message(role=Role.ASSISTANT, content="Assistant"),
        ]

        provider.chat(messages)

        call_args = mock_client.messages.create.call_args
        claude_messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        assert len(claude_messages) == 2
        assert claude_messages[0]["role"] == "user"
        assert claude_messages[1]["role"] == "assistant"


class TestDeepSeekProvider:
    def test_requires_api_key(self) -> None:
        with pytest.raises(ValueError, match="DeepSeek API key is required"):
            DeepSeekProvider(api_key=None, model="test-model")

    def test_requires_api_key_empty(self) -> None:
        with pytest.raises(ValueError, match="DeepSeek API key is required"):
            DeepSeekProvider(api_key="", model="test-model")

    @patch("liz.infrastructure.ai.deepseek.OpenAI")
    def test_creates_client(self, mock_client_class: MagicMock) -> None:
        provider = DeepSeekProvider(api_key="test-key", model="deepseek-chat")
        mock_client_class.assert_called_once_with(
            api_key="test-key",
            base_url="https://api.deepseek.com",
        )
        assert provider.model == "deepseek-chat"

    @patch("liz.infrastructure.ai.deepseek.OpenAI")
    def test_chat_returns_response(self, mock_client_class: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello from DeepSeek"))]
        mock_response.model = "deepseek-chat"
        mock_response.usage.prompt_token_count = 10
        mock_response.usage.completion_token_count = 5
        mock_client.chat.completions.create.return_value = mock_response

        provider = DeepSeekProvider(api_key="test-key", model="deepseek-chat")
        messages = [
            Message(role=Role.SYSTEM, content="You are helpful"),
            Message(role=Role.USER, content="Hi"),
        ]

        result = provider.chat(messages)

        assert isinstance(result, AIResponse)
        assert result.content == "Hello from DeepSeek"
        assert result.model == "deepseek-chat"
        assert result.usage["input_tokens"] == 10
        assert result.usage["output_tokens"] == 5

    @patch("liz.infrastructure.ai.deepseek.OpenAI")
    def test_chat_formats_messages(self, mock_client_class: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Response"))]
        mock_response.model = "deepseek-chat"
        mock_response.usage = None
        mock_client.chat.completions.create.return_value = mock_response

        provider = DeepSeekProvider(api_key="test-key", model="deepseek-chat")
        messages = [
            Message(role=Role.SYSTEM, content="System"),
            Message(role=Role.USER, content="User"),
            Message(role=Role.ASSISTANT, content="Assistant"),
        ]

        provider.chat(messages)

        call_args = mock_client.chat.completions.create.call_args
        ds_messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        assert len(ds_messages) == 3
        assert ds_messages[0]["role"] == "system"
        assert ds_messages[1]["role"] == "user"
        assert ds_messages[2]["role"] == "assistant"
