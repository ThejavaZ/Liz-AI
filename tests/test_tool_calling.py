import pytest
from unittest.mock import patch, MagicMock
from typing import Any

from liz.core.agent import Agent
from liz.core.models import AIResponse, Message, Role, ToolCall
from liz.config.settings import Settings
from liz.tools.registry import ToolRegistry
from liz.tools.permissions import PermissionLayer, Permission
from liz.tools.filesystem import ListDirectory, ReadFile, FileExists
from liz.tools.git import GitStatus, GitLog, GitBranch
from liz.tools.terminal import RunCommand
from liz.tools.models import ToolStatus


def _make_settings(**kwargs) -> Settings:
    return Settings(
        ai_provider="gemini",
        ai_model="test-model",
        gemini_api_key="test-key",
        **kwargs,
    )


def _make_registry(**kwargs) -> ToolRegistry:
    permissions = PermissionLayer()
    permissions.set_default(Permission.ALLOW)
    registry = ToolRegistry(permissions=permissions)
    registry.register(ListDirectory())
    registry.register(ReadFile())
    registry.register(FileExists())
    registry.register(GitStatus())
    registry.register(GitLog())
    registry.register(GitBranch())
    registry.register(RunCommand())
    return registry


class MockToolProvider:
    def __init__(self, responses: list[AIResponse]) -> None:
        self.responses = responses
        self.call_count = 0
        self.call_args_list: list[dict] = []

    def chat(self, messages: list[Message], tools: list[dict[str, Any]] | None = None) -> AIResponse:
        self.call_args_list.append({"messages": messages, "tools": tools})
        response = self.responses[self.call_count]
        self.call_count += 1
        return response


class TestToolDiscovery:
    def test_registry_provides_schemas(self) -> None:
        registry = _make_registry()
        schemas = registry.get_tool_schemas()

        assert len(schemas) == 7
        names = [s["name"] for s in schemas]
        assert "list_directory" in names
        assert "read_file" in names
        assert "file_exists" in names
        assert "git_status" in names
        assert "git_log" in names
        assert "git_branch" in names
        assert "run_command" in names

    def test_schema_has_description(self) -> None:
        registry = _make_registry()
        schemas = registry.get_tool_schemas()

        for schema in schemas:
            assert "description" in schema
            assert len(schema["description"]) > 0

    def test_schema_has_parameters(self) -> None:
        registry = _make_registry()
        schemas = registry.get_tool_schemas()

        for schema in schemas:
            assert "parameters" in schema
            assert "type" in schema["parameters"]


class TestAgentToolCall:
    def test_agent_handles_tool_call(self, tmp_path) -> None:
        responses = [
            AIResponse(
                content="",
                model="test",
                tool_calls=[ToolCall(id="call_1", name="list_directory", arguments={"path": str(tmp_path)})],
            ),
            AIResponse(content="The directory contains files.", model="test"),
        ]
        provider = MockToolProvider(responses)

        with patch("liz.core.agent.create_provider", return_value=provider):
            registry = _make_registry()
            agent = Agent(_make_settings(), "sys", tool_registry=registry)
            result = agent.chat("What files are here?")

            assert result == "The directory contains files."
            assert provider.call_count == 2

    def test_agent_passes_tools_to_provider(self, tmp_path) -> None:
        responses = [
            AIResponse(content="Done.", model="test"),
        ]
        provider = MockToolProvider(responses)

        with patch("liz.core.agent.create_provider", return_value=provider):
            registry = _make_registry()
            agent = Agent(_make_settings(), "sys", tool_registry=registry)
            agent.chat("Hello")

            tools = provider.call_args_list[0]["tools"]
            assert tools is not None
            assert len(tools) == 7

    def test_agent_no_tools_without_registry(self) -> None:
        responses = [
            AIResponse(content="Hello!", model="test"),
        ]
        provider = MockToolProvider(responses)

        with patch("liz.core.agent.create_provider", return_value=provider):
            agent = Agent(_make_settings(), "sys")
            result = agent.chat("Hi")

            assert result == "Hello!"
            tools = provider.call_args_list[0]["tools"]
            assert tools is None


class TestToolExecution:
    def test_execute_tool_success(self, tmp_path) -> None:
        responses = [
            AIResponse(
                content="",
                model="test",
                tool_calls=[ToolCall(id="call_1", name="file_exists", arguments={"path": str(tmp_path / "test.txt")})],
            ),
            AIResponse(content="The file does not exist.", model="test"),
        ]
        provider = MockToolProvider(responses)

        with patch("liz.core.agent.create_provider", return_value=provider):
            registry = _make_registry()
            agent = Agent(_make_settings(), "sys", tool_registry=registry)
            result = agent.chat("Does test.txt exist?")

            assert "not exist" in result.lower() or "does not exist" in result.lower()

    def test_execute_tool_unknown(self) -> None:
        responses = [
            AIResponse(
                content="",
                model="test",
                tool_calls=[ToolCall(id="call_1", name="unknown_tool", arguments={})],
            ),
            AIResponse(content="I cannot do that.", model="test"),
        ]
        provider = MockToolProvider(responses)

        with patch("liz.core.agent.create_provider", return_value=provider):
            registry = _make_registry()
            agent = Agent(_make_settings(), "sys", tool_registry=registry)
            result = agent.chat("Do something unknown")

            assert result == "I cannot do that."


class TestToolPermissions:
    def test_deny_blocks_execution(self) -> None:
        permissions = PermissionLayer()
        permissions.set_rule("list_directory", Permission.DENY)
        registry = ToolRegistry(permissions=permissions)
        registry.register(ListDirectory())

        result = registry.execute("list_directory", path=".")
        assert result.status == ToolStatus.DENIED

    def test_ask_blocks_by_default(self) -> None:
        permissions = PermissionLayer()
        permissions.set_rule("list_directory", Permission.ASK)
        registry = ToolRegistry(permissions=permissions)
        registry.register(ListDirectory())

        result = registry.execute("list_directory", path=".")
        assert result.status == ToolStatus.DENIED

    def test_allow_executes(self, tmp_path) -> None:
        permissions = PermissionLayer()
        permissions.set_rule("list_directory", Permission.ALLOW)
        registry = ToolRegistry(permissions=permissions)
        registry.register(ListDirectory())

        result = registry.execute("list_directory", path=str(tmp_path))
        assert result.status == ToolStatus.SUCCESS


class TestToolResultInAgent:
    def test_tool_error_returns_to_model(self) -> None:
        responses = [
            AIResponse(
                content="",
                model="test",
                tool_calls=[ToolCall(id="call_1", name="read_file", arguments={"path": "/nonexistent/file.txt"})],
            ),
            AIResponse(content="The file was not found.", model="test"),
        ]
        provider = MockToolProvider(responses)

        with patch("liz.core.agent.create_provider", return_value=provider):
            registry = _make_registry()
            agent = Agent(_make_settings(), "sys", tool_registry=registry)
            result = agent.chat("Read the file")

            assert result == "The file was not found."

    def test_tool_denied_returns_to_model(self) -> None:
        responses = [
            AIResponse(
                content="",
                model="test",
                tool_calls=[ToolCall(id="call_1", name="list_directory", arguments={"path": "."})],
            ),
            AIResponse(content="Permission denied.", model="test"),
        ]
        provider = MockToolProvider(responses)

        permissions = PermissionLayer()
        permissions.set_rule("list_directory", Permission.DENY)

        with patch("liz.core.agent.create_provider", return_value=provider):
            registry = ToolRegistry(permissions=permissions)
            registry.register(ListDirectory())
            agent = Agent(_make_settings(), "sys", tool_registry=registry)
            result = agent.chat("List files")

            assert result == "Permission denied."


class TestMultipleToolCalls:
    def test_multiple_tool_calls(self, tmp_path) -> None:
        responses = [
            AIResponse(
                content="",
                model="test",
                tool_calls=[
                    ToolCall(id="call_1", name="file_exists", arguments={"path": str(tmp_path)}),
                    ToolCall(id="call_2", name="list_directory", arguments={"path": str(tmp_path)}),
                ],
            ),
            AIResponse(content="The directory exists and has files.", model="test"),
        ]
        provider = MockToolProvider(responses)

        with patch("liz.core.agent.create_provider", return_value=provider):
            registry = _make_registry()
            agent = Agent(_make_settings(), "sys", tool_registry=registry)
            result = agent.chat("Check the directory")

            assert result == "The directory exists and has files."
