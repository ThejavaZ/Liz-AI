"""Gemini provider tool loop (Bug 2).

The tool loop broke after round one: the live message was
`contents[-1].parts[0].text`, which is None once the last entry is a
function_response; the function_response name was hard-coded to
"tool_result"; assistant function_call parts vanished from the replayed
history; and two calls to the same tool collided on one id.
"""

from typing import Any
from unittest.mock import MagicMock, patch

from google.genai import types

from liz.core.models import AIResponse, Message, Role, ToolCall
from liz.infrastructure.ai.gemini import GeminiProvider


def _make_provider() -> GeminiProvider:
    with patch("liz.infrastructure.ai.gemini.genai.Client"):
        return GeminiProvider(api_key="test-key", model="gemini-test")


def _response(
    parts: list[types.Part] | None = None,
) -> MagicMock:
    response = MagicMock()
    response.candidates = [MagicMock()]
    response.candidates[0].content = MagicMock()
    response.candidates[0].content.parts = parts or []
    response.usage_metadata.prompt_token_count = 1
    response.usage_metadata.candidates_token_count = 1
    return response


def _text_response(text: str) -> MagicMock:
    return _response([types.Part.from_text(text=text)])


def _function_call_response(name: str, args: dict[str, Any]) -> MagicMock:
    return _response([types.Part.from_function_call(name=name, args=args)])


class TestHistoryReplay:
    def test_assistant_tool_calls_reach_history_as_function_calls(
        self,
    ) -> None:
        provider = _make_provider()
        chat = MagicMock()
        chat.send_message.return_value = _text_response("done")
        provider.client.chats.create.return_value = chat
        messages = [
            Message(role=Role.USER, content="list files"),
            Message(
                role=Role.ASSISTANT,
                content="Listing.",
                tool_calls=[
                    ToolCall(
                        id="call_1_list_directory",
                        name="list_directory",
                        arguments={"path": "."},
                    )
                ],
            ),
            Message(
                role=Role.TOOL,
                content="a.txt",
                tool_call_id="call_1_list_directory",
            ),
        ]

        provider.chat(messages)

        history = provider.client.chats.create.call_args.kwargs["history"]
        assistant = history[1]
        assert assistant.role == "model"
        call_part = assistant.parts[1]
        assert call_part.function_call.name == "list_directory"
        assert dict(call_part.function_call.args) == {"path": "."}

    def test_tool_response_uses_real_function_name(self) -> None:
        provider = _make_provider()
        chat = MagicMock()
        chat.send_message.return_value = _text_response("done")
        provider.client.chats.create.return_value = chat
        messages = [
            Message(role=Role.USER, content="list files"),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1_list_directory",
                        name="list_directory",
                        arguments={},
                    )
                ],
            ),
            Message(
                role=Role.TOOL,
                content="a.txt",
                tool_call_id="call_1_list_directory",
            ),
        ]

        provider.chat(messages)

        # the tool result is the live message (last content), not history
        sent = chat.send_message.call_args.args[0]
        assert sent[0].function_response.name == "list_directory"
        assert sent[0].function_response.response == {"result": "a.txt"}
        # and the history keeps the matching function_call
        history = provider.client.chats.create.call_args.kwargs["history"]
        assert history[1].parts[0].function_call.name == "list_directory"

    def test_unknown_tool_call_id_falls_back(self) -> None:
        provider = _make_provider()
        chat = MagicMock()
        chat.send_message.return_value = _text_response("done")
        provider.client.chats.create.return_value = chat
        messages = [
            Message(
                role=Role.TOOL, content="x", tool_call_id="orphan_id"
            ),
        ]

        provider.chat(messages)

        sent = chat.send_message.call_args.args[0]
        assert sent[0].function_response.name == "tool_result"

    def test_empty_assistant_text_with_calls_has_no_text_part(
        self,
    ) -> None:
        provider = _make_provider()
        chat = MagicMock()
        chat.send_message.return_value = _text_response("done")
        provider.client.chats.create.return_value = chat
        messages = [
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(id="c1", name="pwd", arguments={})
                ],
            ),
        ]

        provider.chat(messages)

        sent = chat.send_message.call_args.args[0]
        assert [p.function_call.name for p in sent] == ["pwd"]


class TestLiveMessage:
    def test_last_message_is_parts_not_text(self) -> None:
        """After a tool round the last entry is a function_response whose
        .text is None; the provider must forward the parts, not the text."""
        provider = _make_provider()
        chat = MagicMock()
        chat.send_message.return_value = _text_response("done")
        provider.client.chats.create.return_value = chat
        messages = [
            Message(role=Role.USER, content="go"),
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[ToolCall(id="c1", name="pwd", arguments={})],
            ),
            Message(role=Role.TOOL, content="/home", tool_call_id="c1"),
        ]

        provider.chat(messages)

        sent = chat.send_message.call_args.args[0]
        assert isinstance(sent, list)
        assert all(isinstance(p, types.Part) for p in sent)
        assert sent[0].function_response.name == "pwd"

    def test_plain_user_message_sent_as_parts(self) -> None:
        provider = _make_provider()
        chat = MagicMock()
        chat.send_message.return_value = _text_response("hi")
        provider.client.chats.create.return_value = chat

        provider.chat([Message(role=Role.USER, content="hello")])

        sent = chat.send_message.call_args.args[0]
        assert isinstance(sent, list)
        assert sent[0].text == "hello"

    def test_empty_messages_send_empty_string(self) -> None:
        provider = _make_provider()
        chat = MagicMock()
        chat.send_message.return_value = _text_response("hi")
        provider.client.chats.create.return_value = chat

        provider.chat([])

        assert chat.send_message.call_args.args[0] == ""


class TestUniqueIds:
    def test_parallel_calls_to_same_tool_get_unique_ids(self) -> None:
        provider = _make_provider()
        provider.client.chats.create.return_value = MagicMock()
        response = _response([
            types.Part.from_function_call(name="pwd", args={}),
            types.Part.from_function_call(name="pwd", args={"x": 1}),
        ])

        calls = provider._extract_tool_calls(response)

        assert len(calls) == 2
        assert calls[0].id != calls[1].id
        assert "pwd" in calls[0].id

    def test_ids_unique_across_responses(self) -> None:
        provider = _make_provider()
        provider.client.chats.create.return_value = MagicMock()
        first = _function_call_response("pwd", {})
        second = _function_call_response("pwd", {})

        calls_a = provider._extract_tool_calls(first)
        calls_b = provider._extract_tool_calls(second)

        assert calls_a[0].id != calls_b[0].id

    def test_non_dict_args_become_empty_dict(self) -> None:
        provider = _make_provider()
        provider.client.chats.create.return_value = MagicMock()
        bad_part = types.Part.from_function_call(name="pwd", args={})
        bad_part.function_call.args = None  # type: ignore[union-attr]
        response = _response([bad_part])

        calls = provider._extract_tool_calls(response)

        assert calls[0].arguments == {}


class TestStreamingDedup:
    def test_repeated_call_across_chunks_is_deduped(self) -> None:
        provider = _make_provider()
        chat = MagicMock()
        part = types.Part.from_function_call(name="pwd", args={"a": 1})
        chat.send_message_stream.return_value = iter([
            _response([part]),
            _response([part]),
        ])
        provider.client.chats.create.return_value = chat

        chunks = list(
            provider.stream_chat([Message(role=Role.USER, content="go")])
        )

        final = chunks[-1]
        assert final.done is True
        assert final.tool_calls is not None
        assert len(final.tool_calls) == 1

    def test_distinct_calls_are_kept(self) -> None:
        provider = _make_provider()
        chat = MagicMock()
        chat.send_message_stream.return_value = iter([
            _response([
                types.Part.from_function_call(name="pwd", args={}),
            ]),
            _response([
                types.Part.from_function_call(
                    name="list_directory", args={"path": "."}
                ),
            ]),
        ])
        provider.client.chats.create.return_value = chat

        chunks = list(
            provider.stream_chat([Message(role=Role.USER, content="go")])
        )

        final = chunks[-1]
        assert [c.name for c in final.tool_calls] == [
            "pwd",
            "list_directory",
        ]
        assert final.tool_calls[0].id != final.tool_calls[1].id


class TestFullToolRound:
    def test_round_trip_preserves_pairing(self) -> None:
        """assistant(function_call) -> tool result -> next request must
        produce a history where the response name matches the call."""
        provider = _make_provider()
        chat = MagicMock()
        provider.client.chats.create.return_value = chat
        chat.send_message.side_effect = [
            _function_call_response("list_directory", {"path": "."}),
            _text_response("There is one file."),
        ]
        messages = [Message(role=Role.USER, content="list files")]

        first: AIResponse = provider.chat(messages)
        assert first.tool_calls[0].name == "list_directory"

        follow_up = [
            *messages,
            Message(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        id=first.tool_calls[0].id,
                        name="list_directory",
                        arguments={"path": "."},
                    )
                ],
            ),
            Message(
                role=Role.TOOL,
                content="a.txt",
                tool_call_id=first.tool_calls[0].id,
            ),
        ]
        second = provider.chat(follow_up)

        assert second.content == "There is one file."
        history = provider.client.chats.create.call_args.kwargs["history"]
        # history: user, model(function_call); the tool result is live
        assert [c.role for c in history] == ["user", "model"]
        call_part = history[1].parts[0]
        live = chat.send_message.call_args.args[0]
        response_part = live[0]
        assert (
            response_part.function_response.name
            == call_part.function_call.name
        )
