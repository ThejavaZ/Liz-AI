import json
from unittest.mock import MagicMock, patch

import pytest

from liz.core.models import Message, Role, ToolCall
from liz.infrastructure.ai.openai import OpenAIProvider

TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate math",
        "parameters": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    }
]


def _provider() -> OpenAIProvider:
    with patch("liz.infrastructure.ai.openai.OpenAI"):
        return OpenAIProvider(api_key="k", model="gpt-4o-mini")


class TestConvertTools:
    def test_wraps_as_function(self) -> None:
        converted = OpenAIProvider._convert_tools(TOOLS)

        assert converted[0]["type"] == "function"
        assert converted[0]["function"]["name"] == "calculator"
        assert converted[0]["function"]["parameters"]["type"] == "object"

    def test_includes_description(self) -> None:
        converted = OpenAIProvider._convert_tools(TOOLS)
        assert converted[0]["function"]["description"] == "Evaluate math"

    def test_empty_properties_default(self) -> None:
        converted = OpenAIProvider._convert_tools(
            [{"name": "x", "description": "d"}]
        )
        assert converted[0]["function"]["parameters"]["properties"] == {}


class TestConvertMessages:
    def test_basic_roles(self) -> None:
        provider = _provider()
        result = provider._convert_messages([
            Message(role=Role.SYSTEM, content="sys"),
            Message(role=Role.USER, content="hi"),
        ])

        assert result[0] == {"role": "system", "content": "sys"}
        assert result[1] == {"role": "user", "content": "hi"}

    def test_assistant_with_tool_calls(self) -> None:
        provider = _provider()
        call = ToolCall(id="c1", name="calculator", arguments={"expression": "2+2"})
        result = provider._convert_messages([
            Message(role=Role.ASSISTANT, content="", tool_calls=[call]),
        ])

        entry = result[0]
        assert entry["content"] is None
        assert entry["tool_calls"][0]["id"] == "c1"
        assert entry["tool_calls"][0]["function"]["name"] == "calculator"
        assert json.loads(entry["tool_calls"][0]["function"]["arguments"]) == {
            "expression": "2+2"
        }

    def test_assistant_without_tool_calls(self) -> None:
        provider = _provider()
        result = provider._convert_messages([
            Message(role=Role.ASSISTANT, content="plain answer"),
        ])
        assert "tool_calls" not in result[0]
        assert result[0]["content"] == "plain answer"

    def test_tool_result_message(self) -> None:
        provider = _provider()
        result = provider._convert_messages([
            Message(role=Role.TOOL, content="4", tool_call_id="c1"),
        ])

        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "c1"
        assert result[0]["content"] == "4"

    def test_tool_result_without_id(self) -> None:
        provider = _provider()
        result = provider._convert_messages([
            Message(role=Role.TOOL, content="4"),
        ])
        assert result[0]["tool_call_id"] == ""


class TestParseToolCalls:
    def test_parses_function_call(self) -> None:
        call = MagicMock()
        call.id = "c1"
        call.function.name = "calculator"
        call.function.arguments = '{"expression": "2+2"}'

        result = OpenAIProvider._parse_tool_calls(MagicMock(tool_calls=[call]))

        assert result[0].id == "c1"
        assert result[0].name == "calculator"
        assert result[0].arguments == {"expression": "2+2"}

    def test_no_calls(self) -> None:
        message = MagicMock()
        message.tool_calls = []
        assert OpenAIProvider._parse_tool_calls(message) == []

    def test_none_calls(self) -> None:
        message = MagicMock()
        message.tool_calls = None
        assert OpenAIProvider._parse_tool_calls(message) == []

    def test_bad_json_arguments(self) -> None:
        call = MagicMock()
        call.id = "c1"
        call.function.name = "x"
        call.function.arguments = "{not json"

        result = OpenAIProvider._parse_tool_calls(MagicMock(tool_calls=[call]))
        assert result[0].arguments == {}

    def test_missing_id_generated(self) -> None:
        call = MagicMock()
        call.id = ""
        call.function.name = "x"
        call.function.arguments = "{}"

        result = OpenAIProvider._parse_tool_calls(MagicMock(tool_calls=[call]))
        assert result[0].id == "call_0"


class TestChatSendsTools:
    @patch("liz.infrastructure.ai.openai.OpenAI")
    def test_tools_passed_to_api(self, mock_client_class) -> None:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="hi", tool_calls=[]))]
        response.model = "m"
        response.usage = None
        mock_client.chat.completions.create.return_value = response

        provider = OpenAIProvider(api_key="k", model="m")
        provider.chat([Message(role=Role.USER, content="hi")], tools=TOOLS)

        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["tools"][0]["function"]["name"] == "calculator"

    @patch("liz.infrastructure.ai.openai.OpenAI")
    def test_no_tools_key_when_none(self, mock_client_class) -> None:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="hi", tool_calls=[]))]
        response.model = "m"
        response.usage = None
        mock_client.chat.completions.create.return_value = response

        provider = OpenAIProvider(api_key="k", model="m")
        provider.chat([Message(role=Role.USER, content="hi")])

        _, kwargs = mock_client.chat.completions.create.call_args
        assert "tools" not in kwargs

    @patch("liz.infrastructure.ai.openai.OpenAI")
    def test_tool_calls_parsed_from_response(self, mock_client_class) -> None:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        call = MagicMock()
        call.id = "c1"
        call.function.name = "calculator"
        call.function.arguments = '{"expression": "2+2"}'

        message = MagicMock()
        message.content = None
        message.tool_calls = [call]

        response = MagicMock()
        response.choices = [MagicMock(message=message)]
        response.model = "m"
        response.usage = None
        mock_client.chat.completions.create.return_value = response

        provider = OpenAIProvider(api_key="k", model="m")
        result = provider.chat([Message(role=Role.USER, content="hi")], tools=TOOLS)

        assert result.tool_calls[0].name == "calculator"
        assert result.tool_calls[0].arguments == {"expression": "2+2"}

    @patch("liz.infrastructure.ai.openai.OpenAI")
    def test_stream_sends_tools(self, mock_client_class) -> None:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        chunk = MagicMock()
        chunk.model = "m"
        chunk.usage = None
        choice = MagicMock()
        choice.delta.content = "hi"
        choice.delta.tool_calls = None
        chunk.choices = [choice]
        mock_client.chat.completions.create.return_value = [chunk]

        provider = OpenAIProvider(api_key="k", model="m")
        list(provider.stream_chat([Message(role=Role.USER, content="hi")], tools=TOOLS))

        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["tools"][0]["function"]["name"] == "calculator"


class TestToolCallRoundTrip:
    """A tool call made by the model must survive into the next request."""

    def test_history_contains_call_and_result(self) -> None:
        provider = _provider()

        call = ToolCall(id="c1", name="calculator", arguments={"expression": "2+2"})
        history = [
            Message(role=Role.SYSTEM, content="sys"),
            Message(role=Role.USER, content="what is 2+2"),
            Message(role=Role.ASSISTANT, content="", tool_calls=[call]),
            Message(role=Role.TOOL, content="4", tool_call_id="c1"),
        ]

        converted = provider._convert_messages(history)

        assert converted[2]["tool_calls"][0]["id"] == "c1"
        assert converted[3]["role"] == "tool"
        assert converted[3]["tool_call_id"] == "c1"


class TestContextTrimSafety:
    def test_trim_drops_orphan_tool_result(self) -> None:
        from liz.core.context import Context

        context = Context("sys", max_messages=4)
        call = ToolCall(id="c1", name="x", arguments={})

        context.add_user_message("u1")
        context.add_assistant_message("", tool_calls=[call])
        context.add_tool_result("c1", "r1")
        context.add_user_message("u2")
        context.add_assistant_message("a2")
        context.add_user_message("u3")

        messages = context.get_messages()
        assert messages[0].role == Role.SYSTEM
        assert not any(m.role == Role.TOOL for m in messages)

    def test_trim_keeps_valid_pair(self) -> None:
        from liz.core.context import Context

        context = Context("sys", max_messages=5)
        call = ToolCall(id="c1", name="x", arguments={})

        context.add_user_message("u1")
        context.add_assistant_message("", tool_calls=[call])
        context.add_tool_result("c1", "r1")
        context.add_assistant_message("a1")

        messages = context.get_messages()
        roles = [m.role for m in messages]
        assert Role.TOOL in roles
        tool_msg = next(m for m in messages if m.role == Role.TOOL)
        assert tool_msg.tool_call_id == "c1"
