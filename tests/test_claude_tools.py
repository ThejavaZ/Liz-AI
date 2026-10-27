"""Claude provider tool-calling (Bug 1).

Before this, ClaudeProvider silently dropped tools: schemas were never
sent, `tool_use` blocks were discarded, and `Role.TOOL` messages had no
branch at all. With AI_PROVIDER=claude no tool ever executed.
"""

from typing import Any
from unittest.mock import MagicMock, patch

from liz.core.models import AIResponse, Message, Role, StreamChunk
from liz.infrastructure.ai.claude import ClaudeProvider


def _make_provider() -> ClaudeProvider:
    with patch("liz.infrastructure.ai.claude.anthropic.Anthropic"):
        return ClaudeProvider(api_key="test-key", model="claude-3")


def _text_block(text: str) -> MagicMock:
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _tool_use_block(
    block_id: str, name: str, arguments: Any
) -> MagicMock:
    block = MagicMock()
    block.type = "tool_use"
    block.id = block_id
    block.name = name
    block.input = arguments
    return block


class TestToolSchemaConversion:
    def test_tools_are_sent_with_input_schema(self) -> None:
        provider = _make_provider()
        provider.client.messages.create.return_value = MagicMock(
            content=[_text_block("hi")],
            model="claude-3",
            usage=MagicMock(input_tokens=1, output_tokens=1),
        )
        tools = [{
            "name": "list_directory",
            "description": "List files",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        }]

        provider.chat([Message(role=Role.USER, content="hi")], tools=tools)

        kwargs = provider.client.messages.create.call_args.kwargs
        assert kwargs["tools"] == [{
            "name": "list_directory",
            "description": "List files",
            "input_schema": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        }]

    def test_no_tools_key_when_no_tools(self) -> None:
        provider = _make_provider()
        provider.client.messages.create.return_value = MagicMock(
            content=[_text_block("hi")],
            model="claude-3",
            usage=MagicMock(input_tokens=1, output_tokens=1),
        )

        provider.chat([Message(role=Role.USER, content="hi")])

        kwargs = provider.client.messages.create.call_args.kwargs
        assert "tools" not in kwargs


class TestToolCallParsing:
    def test_tool_use_blocks_become_tool_calls(self) -> None:
        provider = _make_provider()
        provider.client.messages.create.return_value = MagicMock(
            content=[
                _text_block("Let me check."),
                _tool_use_block("toolu_1", "list_directory", {"path": "."}),
            ],
            model="claude-3",
            usage=MagicMock(input_tokens=1, output_tokens=1),
        )

        response = provider.chat([Message(role=Role.USER, content="hi")])

        assert response.content == "Let me check."
        assert len(response.tool_calls) == 1
        call = response.tool_calls[0]
        assert call.id == "toolu_1"
        assert call.name == "list_directory"
        assert call.arguments == {"path": "."}

    def test_non_dict_input_becomes_empty_arguments(self) -> None:
        provider = _make_provider()
        provider.client.messages.create.return_value = MagicMock(
            content=[_tool_use_block("toolu_1", "list_directory", None)],
            model="claude-3",
            usage=MagicMock(input_tokens=1, output_tokens=1),
        )

        response = provider.chat([Message(role=Role.USER, content="hi")])

        assert response.tool_calls[0].arguments == {}

    def test_text_blocks_only_ignored_for_tools(self) -> None:
        provider = _make_provider()
        provider.client.messages.create.return_value = MagicMock(
            content=[_text_block("just text")],
            model="claude-3",
            usage=MagicMock(input_tokens=1, output_tokens=1),
        )

        response = provider.chat([Message(role=Role.USER, content="hi")])

        assert response.tool_calls == []


class TestMessageConversion:
    def test_tool_result_message_becomes_tool_result_block(self) -> None:
        provider = _make_provider()
        provider.client.messages.create.return_value = MagicMock(
            content=[_text_block("ok")],
            model="claude-3",
            usage=MagicMock(input_tokens=1, output_tokens=1),
        )
        messages = [
            Message(role=Role.USER, content="list files"),
            Message(
                role=Role.ASSISTANT,
                content="Listing now.",
                tool_calls=[],
            ),
            Message(
                role=Role.TOOL,
                content="file_a.txt\nfile_b.txt",
                tool_call_id="toolu_1",
            ),
        ]

        provider.chat(messages)

        sent = provider.client.messages.create.call_args.kwargs["messages"]
        assert sent[-1]["role"] == "user"
        assert sent[-1]["content"] == [{
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": "file_a.txt\nfile_b.txt",
        }]

    def test_consecutive_tool_results_merge_into_one_user_message(
        self,
    ) -> None:
        """The API requires alternating roles; N results must not become
        N consecutive user messages."""
        provider = _make_provider()
        provider.client.messages.create.return_value = MagicMock(
            content=[_text_block("ok")],
            model="claude-3",
            usage=MagicMock(input_tokens=1, output_tokens=1),
        )
        messages = [
            Message(role=Role.USER, content="go"),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    type("TC", (), {"id": "a", "name": "t1", "arguments": {}})(),
                    type("TC", (), {"id": "b", "name": "t2", "arguments": {}})(),
                ],
            ),
            Message(role=Role.TOOL, content="r1", tool_call_id="a"),
            Message(role=Role.TOOL, content="r2", tool_call_id="b"),
        ]

        provider.chat(messages)

        sent = provider.client.messages.create.call_args.kwargs["messages"]
        assert sent[-1]["role"] == "user"
        assert len(sent[-1]["content"]) == 2
        assert sent[-1]["content"][0]["tool_use_id"] == "a"
        assert sent[-1]["content"][1]["tool_use_id"] == "b"
        # exactly one user entry for the two results
        roles = [m["role"] for m in sent]
        assert roles == ["user", "assistant", "user"]

    def test_assistant_tool_calls_are_replayed(self) -> None:
        from liz.core.models import ToolCall

        provider = _make_provider()
        provider.client.messages.create.return_value = MagicMock(
            content=[_text_block("ok")],
            model="claude-3",
            usage=MagicMock(input_tokens=1, output_tokens=1),
        )
        messages = [
            Message(role=Role.USER, content="go"),
            Message(
                role=Role.ASSISTANT,
                content="Checking.",
                tool_calls=[
                    ToolCall(id="toolu_1", name="pwd", arguments={})
                ],
            ),
            Message(role=Role.TOOL, content="/home", tool_call_id="toolu_1"),
        ]

        provider.chat(messages)

        sent = provider.client.messages.create.call_args.kwargs["messages"]
        assistant_blocks = sent[1]["content"]
        assert [b["type"] for b in assistant_blocks] == ["text", "tool_use"]
        assert assistant_blocks[1] == {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "pwd",
            "input": {},
        }

    def test_empty_assistant_text_with_tool_calls_is_fine(self) -> None:
        from liz.core.models import ToolCall

        provider = _make_provider()
        provider.client.messages.create.return_value = MagicMock(
            content=[_text_block("ok")],
            model="claude-3",
            usage=MagicMock(input_tokens=1, output_tokens=1),
        )
        messages = [
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(id="t1", name="pwd", arguments={})
                ],
            ),
        ]

        provider.chat(messages)

        sent = provider.client.messages.create.call_args.kwargs["messages"]
        # no empty text block before the tool_use block
        assert [b["type"] for b in sent[0]["content"]] == ["tool_use"]

    def test_empty_tool_result_content_is_replaced(self) -> None:
        provider = _make_provider()
        provider.client.messages.create.return_value = MagicMock(
            content=[_text_block("ok")],
            model="claude-3",
            usage=MagicMock(input_tokens=1, output_tokens=1),
        )
        messages = [
            Message(role=Role.TOOL, content="", tool_call_id="t1"),
        ]

        provider.chat(messages)

        sent = provider.client.messages.create.call_args.kwargs["messages"]
        assert sent[-1]["content"][0]["content"] == "(no output)"

    def test_system_message_becomes_system_kwarg(self) -> None:
        provider = _make_provider()
        provider.client.messages.create.return_value = MagicMock(
            content=[_text_block("ok")],
            model="claude-3",
            usage=MagicMock(input_tokens=1, output_tokens=1),
        )
        messages = [
            Message(role=Role.SYSTEM, content="Be helpful"),
            Message(role=Role.USER, content="hi"),
        ]

        provider.chat(messages)

        kwargs = provider.client.messages.create.call_args.kwargs
        assert kwargs["system"] == "Be helpful"
        assert [m["role"] for m in kwargs["messages"]] == ["user"]


class TestStreamingToolCalls:
    def _stream(self, events: list[Any]) -> MagicMock:
        stream_cm = MagicMock()
        stream_cm.__enter__.return_value = events
        provider = _make_provider()
        provider.client.messages.stream.return_value = stream_cm
        return provider

    def _event(self, event_type: str, **kwargs: Any) -> MagicMock:
        event = MagicMock()
        event.type = event_type
        for key, value in kwargs.items():
            setattr(event, key, value)
        return event

    def test_tool_use_events_become_tool_calls(self) -> None:
        block = MagicMock()
        block.type = "tool_use"
        block.id = "toolu_1"
        block.name = "list_directory"
        events = [
            self._event(
                "content_block_start", index=0, content_block=block
            ),
            self._event(
                "content_block_delta",
                index=0,
                delta=MagicMock(
                    type="input_json_delta", partial_json='{"path": "'
                ),
            ),
            self._event(
                "content_block_delta",
                index=0,
                delta=MagicMock(
                    type="input_json_delta", partial_json='."}'
                ),
            ),
            self._event("content_block_stop", index=0),
        ]
        provider = self._stream(events)

        chunks = list(
            provider.stream_chat([Message(role=Role.USER, content="go")])
        )

        final = chunks[-1]
        assert final.done is True
        assert final.tool_calls is not None
        assert final.tool_calls[0].id == "toolu_1"
        assert final.tool_calls[0].name == "list_directory"
        assert final.tool_calls[0].arguments == {"path": "."}

    def test_malformed_partial_json_becomes_empty_arguments(self) -> None:
        block = MagicMock()
        block.type = "tool_use"
        block.id = "toolu_1"
        block.name = "t"
        events = [
            self._event(
                "content_block_start", index=0, content_block=block
            ),
            self._event(
                "content_block_delta",
                index=0,
                delta=MagicMock(
                    type="input_json_delta", partial_json="{broken"
                ),
            ),
        ]
        provider = self._stream(events)

        chunks = list(
            provider.stream_chat([Message(role=Role.USER, content="go")])
        )

        assert chunks[-1].tool_calls[0].arguments == {}

    def test_text_and_tools_in_one_stream(self) -> None:
        block = MagicMock()
        block.type = "tool_use"
        block.id = "toolu_1"
        block.name = "pwd"
        events = [
            self._event(
                "content_block_delta",
                index=0,
                delta=MagicMock(type="text_delta", text="Working"),
            ),
            self._event(
                "content_block_start", index=1, content_block=block
            ),
            self._event(
                "content_block_delta",
                index=1,
                delta=MagicMock(type="input_json_delta", partial_json="{}"),
            ),
        ]
        provider = self._stream(events)

        chunks = list(
            provider.stream_chat([Message(role=Role.USER, content="go")])
        )

        text = "".join(c.text for c in chunks)
        assert text == "Working"
        assert chunks[-1].tool_calls[0].name == "pwd"

    def test_no_tool_events_yields_empty_tool_calls(self) -> None:
        events = [
            self._event(
                "content_block_delta",
                index=0,
                delta=MagicMock(type="text_delta", text="Hello"),
            ),
        ]
        provider = self._stream(events)

        chunks = list(
            provider.stream_chat([Message(role=Role.USER, content="go")])
        )

        assert chunks[-1].tool_calls == []
        assert isinstance(chunks[-1], StreamChunk)


class TestChatToolRoundTrip:
    def test_full_round_preserves_ids(self) -> None:
        """assistant(tool_use) -> tool_result -> next request: the id the
        model sees must match the id it asked with."""
        from liz.core.models import ToolCall

        provider = _make_provider()
        provider.client.messages.create.side_effect = [
            MagicMock(
                content=[_tool_use_block("toolu_9", "pwd", {})],
                model="claude-3",
                usage=MagicMock(input_tokens=1, output_tokens=1),
            ),
            MagicMock(
                content=[_text_block("You are in /home")],
                model="claude-3",
                usage=MagicMock(input_tokens=2, output_tokens=1),
            ),
        ]
        messages = [Message(role=Role.USER, content="where am i")]

        first: AIResponse = provider.chat(messages)
        assert first.tool_calls[0].id == "toolu_9"

        follow_up = [
            *messages,
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(id="toolu_9", name="pwd", arguments={})
                ],
            ),
            Message(role=Role.TOOL, content="/home", tool_call_id="toolu_9"),
        ]
        second = provider.chat(follow_up)

        assert second.content == "You are in /home"
        sent = provider.client.messages.create.call_args.kwargs["messages"]
        tool_result = sent[-1]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "toolu_9"
