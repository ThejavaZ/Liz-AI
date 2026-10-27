"""Bug 3 — a tool must never break the tool_call -> result pairing.

If a tool raises, or a provider emits arguments that are not an object,
the agent must still produce one TOOL result per tool_call. Otherwise
the next provider request fails with a 400 and the context is poisoned
until /clear.
"""

from typing import Any
from unittest.mock import patch

from liz.config.settings import Settings
from liz.core.agent import Agent, _short_args
from liz.core.models import AIResponse, Message, Role, StreamChunk, ToolCall
from liz.infrastructure.ai.base import AIProvider
from liz.tools.base import Tool
from liz.tools.models import ToolResult, ToolStatus
from liz.tools.permissions import Permission, PermissionLayer
from liz.tools.registry import ToolRegistry


def _make_settings(**kwargs) -> Settings:
    return Settings(
        ai_provider="gemini",
        ai_model="test-model",
        gemini_api_key="test-key",
        **kwargs,
    )


class CrashingTool(Tool):
    @property
    def name(self) -> str:
        return "crash_tool"

    @property
    def description(self) -> str:
        return "Always explodes."

    def execute(self, **kwargs) -> ToolResult:
        raise RuntimeError("boom")


class EchoTool(Tool):
    @property
    def name(self) -> str:
        return "echo_tool"

    @property
    def description(self) -> str:
        return "Echoes a value."

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult.success(str(kwargs.get("value", "")))


class StrictTool(Tool):
    """Tool with a fixed signature: unexpected keywords raise TypeError."""

    @property
    def name(self) -> str:
        return "strict_tool"

    @property
    def description(self) -> str:
        return "Accepts exactly one keyword."

    def execute(self, value: str) -> ToolResult:  # type: ignore[override]
        return ToolResult.success(value)


def _make_registry(*tools: Tool) -> ToolRegistry:
    permissions = PermissionLayer()
    permissions.set_default(Permission.ALLOW)
    registry = ToolRegistry(permissions=permissions)
    for tool in tools:
        registry.register(tool)
    return registry


class MockToolProvider:
    """Scripted provider that records the messages of every call."""

    def __init__(self, responses: list[AIResponse]) -> None:
        self.responses = responses
        self.call_count = 0
        self.seen_messages: list[list[Message]] = []

    def chat(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> AIResponse:
        self.seen_messages.append(list(messages))
        response = self.responses[self.call_count]
        self.call_count += 1
        return response


def _tool_results_after(messages: list[Message]) -> list[Message]:
    """Return the TOOL messages of a conversation snapshot."""
    return [m for m in messages if m.role is Role.TOOL]


class TestRegistryCrashHandling:
    def test_tool_exception_becomes_error_result(self) -> None:
        registry = _make_registry(CrashingTool())

        result = registry.execute("crash_tool")

        assert result.status == ToolStatus.ERROR
        assert "boom" in (result.error or "")

    def test_error_names_the_tool(self) -> None:
        registry = _make_registry(CrashingTool())

        result = registry.execute("crash_tool")

        assert "crash_tool" in (result.error or "")

    def test_unexpected_keyword_becomes_error_result(self) -> None:
        registry = _make_registry(StrictTool())

        result = registry.execute("strict_tool", value="ok", bogus=1)

        assert result.status == ToolStatus.ERROR
        assert result.error


class TestAgentSurvivesCrashingTool:
    def test_chat_completes_despite_crash(self) -> None:
        responses = [
            AIResponse(
                content="",
                model="test",
                tool_calls=[
                    ToolCall(id="c1", name="crash_tool", arguments={})
                ],
            ),
            AIResponse(content="That tool is broken.", model="test"),
        ]
        provider = MockToolProvider(responses)

        with patch("liz.core.agent.create_provider", return_value=provider):
            agent = Agent(
                _make_settings(),
                "sys",
                tool_registry=_make_registry(CrashingTool()),
            )
            result = agent.chat("use the crashing tool")

        assert result == "That tool is broken."
        assert provider.call_count == 2

    def test_crash_error_is_fed_back_to_model(self) -> None:
        responses = [
            AIResponse(
                content="",
                model="test",
                tool_calls=[
                    ToolCall(id="c1", name="crash_tool", arguments={})
                ],
            ),
            AIResponse(content="ok", model="test"),
        ]
        provider = MockToolProvider(responses)

        with patch("liz.core.agent.create_provider", return_value=provider):
            agent = Agent(
                _make_settings(),
                "sys",
                tool_registry=_make_registry(CrashingTool()),
            )
            agent.chat("use the crashing tool")

        second_call = provider.seen_messages[1]
        results = _tool_results_after(second_call)
        assert len(results) == 1
        assert results[0].tool_call_id == "c1"
        assert "boom" in results[0].content

    def test_every_tool_call_gets_a_result(self) -> None:
        """N requested calls must yield N results, even mid-round crashes."""
        responses = [
            AIResponse(
                content="",
                model="test",
                tool_calls=[
                    ToolCall(
                        id="c1", name="echo_tool", arguments={"value": "a"}
                    ),
                    ToolCall(id="c2", name="crash_tool", arguments={}),
                    ToolCall(
                        id="c3", name="echo_tool", arguments={"value": "b"}
                    ),
                ],
            ),
            AIResponse(content="done", model="test"),
        ]
        provider = MockToolProvider(responses)

        with patch("liz.core.agent.create_provider", return_value=provider):
            agent = Agent(
                _make_settings(),
                "sys",
                tool_registry=_make_registry(CrashingTool(), EchoTool()),
            )
            result = agent.chat("three tools please")

        assert result == "done"
        second_call = provider.seen_messages[1]
        results = _tool_results_after(second_call)
        assert [m.tool_call_id for m in results] == ["c1", "c2", "c3"]
        # the crashing middle call still produced an error, not silence
        assert "boom" in results[1].content

    def test_context_never_orphans_tool_calls(self) -> None:
        responses = [
            AIResponse(
                content="",
                model="test",
                tool_calls=[
                    ToolCall(id="c1", name="crash_tool", arguments={}),
                    ToolCall(
                        id="c2", name="echo_tool", arguments={"value": "x"}
                    ),
                ],
            ),
            AIResponse(content="done", model="test"),
        ]
        provider = MockToolProvider(responses)

        with patch("liz.core.agent.create_provider", return_value=provider):
            agent = Agent(
                _make_settings(),
                "sys",
                tool_registry=_make_registry(CrashingTool(), EchoTool()),
            )
            agent.chat("two tools")

        messages = agent.context.get_messages()
        pending: set[str] = set()
        for message in messages:
            if message.role is Role.ASSISTANT and message.tool_calls:
                pending.update(tc.id for tc in message.tool_calls)
            elif message.role is Role.TOOL:
                pending.discard(message.tool_call_id)
        assert not pending, f"orphaned tool calls: {pending}"


class TestMalformedArguments:
    def test_non_dict_arguments_become_error_result(self) -> None:
        bad_call = ToolCall(
            id="c1", name="echo_tool", arguments=["not", "a", "dict"]  # type: ignore[arg-type]
        )

        with patch(
            "liz.core.agent.create_provider",
            return_value=MockToolProvider(
                [
                    AIResponse(content="", model="test", tool_calls=[bad_call]),
                    AIResponse(content="ok", model="test"),
                ]
            ),
        ):
            agent = Agent(
                _make_settings(),
                "sys",
                tool_registry=_make_registry(EchoTool()),
            )
            result = agent.chat("weird args")

        assert result == "ok"
        messages = agent.context.get_messages()
        results = _tool_results_after(messages)
        assert len(results) == 1
        assert results[0].tool_call_id == "c1"
        assert "JSON object" in results[0].content

    def test_execute_tool_direct_with_list_arguments(self) -> None:
        agent = Agent.__new__(Agent)
        agent.tool_registry = _make_registry(EchoTool())
        call = ToolCall(
            id="c1", name="echo_tool", arguments=[1, 2]  # type: ignore[arg-type]
        )

        message = agent._execute_tool(call)

        assert message.startswith("Error:")

    def test_short_args_handles_non_dict(self) -> None:
        call = ToolCall(
            id="c1", name="echo_tool", arguments=[1, 2]  # type: ignore[arg-type]
        )

        text = _short_args(call)

        assert "[1, 2]" in text

    def test_short_args_still_truncates(self) -> None:
        call = ToolCall(
            id="c1",
            name="echo_tool",
            arguments={"value": "x" * 200},
        )

        text = _short_args(call, limit=40)

        assert len(text) <= 40


class StreamingCrashProvider(AIProvider):
    """First stream requests a crashing tool, second answers."""

    def __init__(self, tool_calls: list[ToolCall]) -> None:
        self.model = "tool"
        self.calls = 0
        self.tool_calls = tool_calls

    def chat(self, messages, tools=None):
        return AIResponse(content="done", model=self.model)

    def stream_chat(self, messages, tools=None):
        self.calls += 1
        if self.calls == 1:
            yield StreamChunk(
                text="",
                tool_calls=self.tool_calls,
                done=True,
                response=AIResponse(
                    content="", model=self.model, tool_calls=self.tool_calls
                ),
            )
        else:
            yield StreamChunk(text="recovered")
            yield StreamChunk(
                text="",
                done=True,
                response=AIResponse(content="recovered", model=self.model),
            )


class TestStreamingSurvivesCrash:
    def test_stream_completes_despite_crash(self) -> None:
        provider = StreamingCrashProvider(
            [ToolCall(id="c1", name="crash_tool", arguments={})]
        )

        with patch("liz.core.agent.create_provider", return_value=provider):
            agent = Agent(
                _make_settings(),
                "sys",
                tool_registry=_make_registry(CrashingTool()),
            )
            chunks = list(agent.stream_chat("go"))

        assert chunks[-1].done is True
        assert any(c.text == "recovered" for c in chunks)

    def test_tool_events_are_reported(self) -> None:
        provider = StreamingCrashProvider(
            [ToolCall(id="c1", name="crash_tool", arguments={})]
        )

        with patch("liz.core.agent.create_provider", return_value=provider):
            agent = Agent(
                _make_settings(),
                "sys",
                tool_registry=_make_registry(CrashingTool()),
            )
            chunks = list(agent.stream_chat("go"))

        events = [e for c in chunks for e in c.tool_events]
        assert events == ["crash_tool()"]

    def test_stream_context_keeps_pairing(self) -> None:
        provider = StreamingCrashProvider(
            [
                ToolCall(id="c1", name="crash_tool", arguments={}),
                ToolCall(id="c2", name="echo_tool", arguments={"value": "v"}),
            ]
        )

        with patch("liz.core.agent.create_provider", return_value=provider):
            agent = Agent(
                _make_settings(),
                "sys",
                tool_registry=_make_registry(CrashingTool(), EchoTool()),
            )
            list(agent.stream_chat("go"))

        results = _tool_results_after(agent.context.get_messages())
        assert [m.tool_call_id for m in results] == ["c1", "c2"]
        assert "boom" in results[0].content
