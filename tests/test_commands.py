import pytest
from unittest.mock import MagicMock

from liz.cli.commands import CommandHandler, CommandResult
from liz.config.settings import Settings
from liz.core.context import Context
from liz.tools.registry import ToolRegistry
from liz.tools.permissions import PermissionLayer, Permission
from liz.tools.filesystem import ListDirectory, ReadFile, FileExists
from liz.tools.git import GitStatus, GitLog, GitBranch
from liz.tools.terminal import RunCommand


def _make_handler(**kwargs) -> CommandHandler:
    settings = Settings(
        ai_provider="gemini",
        ai_model="test-model",
        gemini_api_key="test-key",
    )
    context = Context("system prompt")
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

    return CommandHandler(settings, context, registry)


class TestCommandHandler:
    def test_returns_none_for_normal_text(self) -> None:
        handler = _make_handler()
        result = handler.handle("Hello world")
        assert result is None

    def test_returns_none_for_non_slash_input(self) -> None:
        handler = _make_handler()
        result = handler.handle("not a command")
        assert result is None

    def test_unknown_command(self) -> None:
        handler = _make_handler()
        result = handler.handle("/unknown")
        assert result is not None
        assert "Unknown command" in result.output
        assert "/unknown" in result.output

    def test_exit_command(self) -> None:
        handler = _make_handler()
        result = handler.handle("/exit")
        assert result is not None
        assert result.should_exit is True
        assert "Goodbye" in result.output

    def test_quit_command(self) -> None:
        handler = _make_handler()
        result = handler.handle("/quit")
        assert result is not None
        assert result.should_exit is True
        assert "Goodbye" in result.output


class TestHelpCommand:
    def test_help_output(self) -> None:
        handler = _make_handler()
        result = handler.handle("/help")
        assert result is not None
        assert result.should_exit is False
        assert "/help" in result.output
        assert "/tools" in result.output
        assert "/clear" in result.output
        assert "/status" in result.output
        assert "/exit" in result.output

    def test_help_case_insensitive(self) -> None:
        handler = _make_handler()
        result = handler.handle("/HELP")
        assert result is not None
        assert "/help" in result.output


class TestToolsCommand:
    def test_tools_lists_registered_tools(self) -> None:
        handler = _make_handler()
        result = handler.handle("/tools")
        assert result is not None
        assert "list_directory" in result.output
        assert "read_file" in result.output
        assert "git_status" in result.output
        assert "run_command" in result.output

    def test_tools_shows_descriptions(self) -> None:
        handler = _make_handler()
        result = handler.handle("/tools")
        assert result is not None
        assert "List files" in result.output
        assert "Read the contents" in result.output


class TestClearCommand:
    def test_clear_resets_context(self) -> None:
        handler = _make_handler()
        handler.context.add_user_message("Hello")
        handler.context.add_assistant_message("Hi")
        assert len(handler.context.get_messages()) == 3

        result = handler.handle("/clear")
        assert result is not None
        assert "cleared" in result.output.lower()
        assert len(handler.context.get_messages()) == 1


class TestStatusCommand:
    def test_status_shows_provider(self) -> None:
        handler = _make_handler()
        result = handler.handle("/status")
        assert result is not None
        assert "gemini" in result.output.lower()

    def test_status_shows_model(self) -> None:
        handler = _make_handler()
        result = handler.handle("/status")
        assert result is not None
        assert "test-model" in result.output

    def test_status_shows_tools_count(self) -> None:
        handler = _make_handler()
        result = handler.handle("/status")
        assert result is not None
        assert "7" in result.output


class TestCommandsDoNotCallAI:
    def test_help_does_not_call_provider(self) -> None:
        handler = _make_handler()
        mock_provider = MagicMock()
        handler._commands["help"] = handler._help
        result = handler.handle("/help")
        assert result is not None

    def test_commands_are_independent(self) -> None:
        handler = _make_handler()
        for cmd in ["/help", "/tools", "/clear", "/status", "/exit", "/quit"]:
            result = handler.handle(cmd)
            assert result is not None


class TestCommandsReflectRegistry:
    def test_tools_reflects_actual_registry(self) -> None:
        permissions = PermissionLayer()
        permissions.set_default(Permission.ALLOW)
        registry = ToolRegistry(permissions=permissions)
        registry.register(ListDirectory())
        registry.register(GitStatus())

        settings = Settings(
            ai_provider="gemini",
            ai_model="test-model",
            gemini_api_key="test-key",
        )
        context = Context("sys")
        handler = CommandHandler(settings, context, registry)

        result = handler.handle("/tools")
        assert "list_directory" in result.output
        assert "git_status" in result.output
        assert "read_file" not in result.output
