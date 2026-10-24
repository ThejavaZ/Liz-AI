from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from liz.config.settings import Settings
from liz.core.agent import Agent
from liz.core.context import Context
from liz.core.memory import ConversationMemory
from liz.core.models import (
    AIResponse,
    Message,
    Role,
    StreamChunk,
    ToolCall,
)
from liz.infrastructure.ai.base import AIProvider
from liz.infrastructure.ai.fallback import FallbackProvider
from liz.infrastructure.ai.openai import OpenAIProvider


class TestStreamChunk:
    def test_defaults(self) -> None:
        chunk = StreamChunk()
        assert chunk.text == ""
        assert chunk.tool_calls == []
        assert chunk.tool_events == []
        assert chunk.done is False
        assert chunk.response is None

    def test_text_chunk(self) -> None:
        chunk = StreamChunk(text="hello", done=False)
        assert chunk.text == "hello"

    def test_final_chunk(self) -> None:
        response = AIResponse(content="full", model="m")
        chunk = StreamChunk(done=True, response=response)
        assert chunk.done is True
        assert chunk.response.content == "full"


class _StreamProvider(AIProvider):
    """Provider that yields predefined chunks."""

    def __init__(
        self,
        chunks: list[StreamChunk],
        error: Exception | None = None,
        model: str = "stream-model",
    ) -> None:
        self.chunks = chunks
        self.error = error
        self.model = model
        self.calls = 0

    def chat(self, messages, tools=None):
        return AIResponse(content="non-stream", model=self.model)

    def stream_chat(self, messages, tools=None):
        self.calls += 1
        if self.error:
            raise self.error
        yield from self.chunks


class TestBaseStreamChat:
    def test_default_yields_single_chunk(self) -> None:
        provider = _StreamProvider([])
        provider.stream_chat = AIProvider.stream_chat.__get__(provider)

        chunks = list(provider.stream_chat([Message(role=Role.USER, content="hi")]))

        assert len(chunks) == 1
        assert chunks[0].text == "non-stream"
        assert chunks[0].done is True

    def test_default_includes_tool_calls(self) -> None:
        provider = _StreamProvider([])
        provider.stream_chat = AIProvider.stream_chat.__get__(provider)
        provider.chat = lambda m, t=None: AIResponse(
            content="",
            model="m",
            tool_calls=[ToolCall(id="1", name="calc", arguments={})],
        )

        chunks = list(provider.stream_chat([]))
        assert chunks[0].tool_calls[0].name == "calc"


class TestFallbackStreamChat:
    def test_yields_chunks_from_first(self) -> None:
        first = _StreamProvider([StreamChunk(text="a"), StreamChunk(text="b")])
        second = _StreamProvider([StreamChunk(text="c")])

        chunks = list(FallbackProvider([first, second]).stream_chat([]))

        assert [c.text for c in chunks] == ["a", "b"]
        assert second.calls == 0

    def test_falls_back_before_any_output(self) -> None:
        first = _StreamProvider([], error=Exception("429 rate limit"))
        second = _StreamProvider([StreamChunk(text="ok")])

        chunks = list(FallbackProvider([first, second]).stream_chat([]))

        assert chunks[0].text == "ok"

    def test_no_fallback_after_output_started(self) -> None:
        first = _StreamProvider(
            [StreamChunk(text="partial")],
            error=None,
        )
        # make it raise mid-stream
        def failing_stream(messages, tools=None):
            yield StreamChunk(text="partial")
            raise Exception("503 unavailable")

        first.stream_chat = failing_stream
        second = _StreamProvider([StreamChunk(text="never")])

        with pytest.raises(Exception, match="503"):
            list(FallbackProvider([first, second]).stream_chat([]))

        assert second.calls == 0

    def test_raises_last_error_when_exhausted(self) -> None:
        first = _StreamProvider([], error=Exception("429"))
        second = _StreamProvider([], error=Exception("503"))

        with pytest.raises(Exception, match="503"):
            list(FallbackProvider([first, second]).stream_chat([]))


class _EchoProvider(AIProvider):
    def __init__(self, content="hello world", tool_calls=None) -> None:
        self.model = "echo"
        self.content = content
        self.tool_calls = tool_calls or []
        self.seen_messages: list[list[Message]] = []

    def chat(self, messages, tools=None):
        self.seen_messages.append(list(messages))
        return AIResponse(
            content=self.content, model=self.model, tool_calls=self.tool_calls
        )

    def stream_chat(self, messages, tools=None):
        self.seen_messages.append(list(messages))
        words = self.content.split(" ")
        for index, word in enumerate(words):
            piece = word if index == 0 else " " + word
            yield StreamChunk(text=piece)

        yield StreamChunk(
            text="",
            tool_calls=self.tool_calls,
            done=True,
            response=AIResponse(
                content=self.content,
                model=self.model,
                tool_calls=self.tool_calls,
            ),
        )


class _ToolProvider(AIProvider):
    """First call requests a tool, second call answers."""

    def __init__(self) -> None:
        self.model = "tool"
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            return AIResponse(
                content="",
                model=self.model,
                tool_calls=[ToolCall(
                    id="c1", name="calculator", arguments={"expression": "2+2"}
                )],
            )
        return AIResponse(content="The answer is 4", model=self.model)

    def stream_chat(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            yield StreamChunk(
                text="",
                tool_calls=[ToolCall(
                    id="c1", name="calculator", arguments={"expression": "2+2"}
                )],
                done=True,
                response=AIResponse(
                    content="",
                    model=self.model,
                    tool_calls=[ToolCall(
                        id="c1",
                        name="calculator",
                        arguments={"expression": "2+2"},
                    )],
                ),
            )
        else:
            yield StreamChunk(text="The answer is 4")
            yield StreamChunk(
                text="",
                done=True,
                response=AIResponse(content="The answer is 4", model=self.model),
            )


class TestAgentStreamChat:
    def _agent(self, provider) -> Agent:
        settings = Settings(ai_provider="gemini", gemini_api_key="k")
        with patch("liz.core.agent.create_provider", return_value=provider):
            return Agent(settings, "sys")

    def test_yields_text_chunks(self) -> None:
        agent = self._agent(_EchoProvider("hello world"))

        chunks = list(agent.stream_chat("hi"))

        text = "".join(c.text for c in chunks)
        assert text == "hello world"
        assert chunks[-1].done is True

    def test_saves_to_context(self) -> None:
        agent = self._agent(_EchoProvider("hello world"))
        list(agent.stream_chat("hi"))

        roles = [m.role for m in agent.context.get_messages()]
        assert Role.USER in roles
        assert Role.ASSISTANT in roles

    def test_saves_to_memory(self) -> None:
        from tests.test_memory import InMemoryStore

        store = InMemoryStore()
        memory = ConversationMemory(store, session_id="s1")
        settings = Settings(ai_provider="gemini", gemini_api_key="k")

        with patch("liz.core.agent.create_provider", return_value=_EchoProvider()):
            agent = Agent(settings, "sys", memory=memory)

        list(agent.stream_chat("hi"))

        history = store.get_history("s1")
        assert len(history) == 2
        assert history[0].role == Role.USER

    def test_executes_tools_and_reports_events(self) -> None:
        from liz.tools.calculator import Calculator
        from liz.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(Calculator())

        provider = _ToolProvider()
        settings = Settings(ai_provider="gemini", gemini_api_key="k")
        with patch("liz.core.agent.create_provider", return_value=provider):
            agent = Agent(settings, "sys", tool_registry=registry)

        chunks = list(agent.stream_chat("what is 2+2"))

        events = [e for c in chunks for e in c.tool_events]
        assert any("calculator" in event for event in events)

        text = "".join(c.text for c in chunks)
        assert "The answer is 4" in text

    def test_final_text_present_after_tools(self) -> None:
        from liz.tools.calculator import Calculator
        from liz.tools.registry import ToolRegistry

        registry = ToolRegistry()
        registry.register(Calculator())

        provider = _ToolProvider()
        settings = Settings(ai_provider="gemini", gemini_api_key="k")
        with patch("liz.core.agent.create_provider", return_value=provider):
            agent = Agent(settings, "sys", tool_registry=registry)

        chunks = list(agent.stream_chat("what is 2+2"))

        assert chunks[-1].done is True
        assert any(c.text for c in chunks)


class TestOpenAIStreamChat:
    def _chunk(self, content=None, tool_calls=None, usage=None, model=None):
        chunk = MagicMock()
        chunk.model = model or "gpt-4o-mini"
        chunk.usage = usage
        choice = MagicMock()
        delta = MagicMock()
        if content is None:
            delta.content = None
        else:
            delta.content = content
        if tool_calls is None:
            delta.tool_calls = None
        else:
            delta.tool_calls = tool_calls
        choice.delta = delta
        chunk.choices = [choice]
        return chunk

    @patch("liz.infrastructure.ai.openai.OpenAI")
    def test_streams_text_deltas(self, mock_client_class) -> None:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = [
            self._chunk(content="Hello"),
            self._chunk(content=" world"),
            self._chunk(content=None, usage=MagicMock(prompt_tokens=5, completion_tokens=2)),
        ]

        provider = OpenAIProvider(api_key="k", model="gpt-4o-mini")
        chunks = list(provider.stream_chat([Message(role=Role.USER, content="hi")]))

        text = "".join(c.text for c in chunks)
        assert text == "Hello world"
        assert chunks[-1].done is True
        assert chunks[-1].response.content == "Hello world"

    @patch("liz.infrastructure.ai.openai.OpenAI")
    def test_accumulates_tool_calls(self, mock_client_class) -> None:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        part1 = MagicMock(id="call_1", index=0)
        part1.function = MagicMock(name="calc", arguments='{"exp')
        part1.function.name = "calc"
        part1.function.arguments = '{"exp'

        part2 = MagicMock(id="", index=0)
        part2.function = MagicMock()
        part2.function.name = None
        part2.function.arguments = 'ression": "2+2"}'

        mock_client.chat.completions.create.return_value = [
            self._chunk(content=None, tool_calls=[part1]),
            self._chunk(content=None, tool_calls=[part2]),
            self._chunk(content=None, usage=None),
        ]

        provider = OpenAIProvider(api_key="k", model="gpt-4o-mini")
        chunks = list(provider.stream_chat([Message(role=Role.USER, content="hi")]))

        final = chunks[-1]
        assert len(final.tool_calls) == 1
        assert final.tool_calls[0].name == "calc"
        assert final.tool_calls[0].arguments == {"expression": "2+2"}

    @patch("liz.infrastructure.ai.openai.OpenAI")
    def test_usage_captured(self, mock_client_class) -> None:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        usage = MagicMock(prompt_tokens=7, completion_tokens=3)
        mock_client.chat.completions.create.return_value = [
            self._chunk(content="hi", usage=usage),
        ]

        provider = OpenAIProvider(api_key="k", model="gpt-4o-mini")
        chunks = list(provider.stream_chat([Message(role=Role.USER, content="x")]))

        assert chunks[-1].response.usage["input_tokens"] == 7
        assert chunks[-1].response.usage["output_tokens"] == 3

    @patch("liz.infrastructure.ai.openai.OpenAI")
    def test_stream_uses_stream_flag(self, mock_client_class) -> None:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = [
            self._chunk(content="x"),
        ]

        provider = OpenAIProvider(api_key="k", model="gpt-4o-mini")
        list(provider.stream_chat([Message(role=Role.USER, content="x")]))

        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["stream"] is True


class TestOllamaStreaming:
    def test_ollama_inherits_stream_chat(self) -> None:
        from liz.infrastructure.ai.ollama import OllamaProvider

        with patch("liz.infrastructure.ai.openai.OpenAI"):
            provider = OllamaProvider(model="llama3.1")

        assert provider.stream_chat.__func__ is OpenAIProvider.stream_chat
