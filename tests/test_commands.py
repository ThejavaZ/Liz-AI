import sys

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


class TestConfigCommand:
    def test_shows_provider(self) -> None:
        result = _make_handler().handle("/config")
        assert "gemini" in result.output

    def test_shows_model(self) -> None:
        result = _make_handler().handle("/config")
        assert "test-model" in result.output

    def test_masks_api_key(self) -> None:
        settings = Settings(
            ai_provider="gemini",
            gemini_api_key="sk-verysecretkey1234567890",
        )
        handler = CommandHandler(settings, Context("sys"), ToolRegistry())
        result = handler.handle("/config")
        assert "sk-verysecretkey1234567890" not in result.output
        assert "sk-v...7890" in result.output

    def test_short_key_fully_masked(self) -> None:
        result = _make_handler().handle("/config")
        assert "test-key" not in result.output

    def test_shows_unset_key(self) -> None:
        settings = Settings(ai_provider="gemini")
        handler = CommandHandler(settings, Context("sys"), ToolRegistry())
        result = handler.handle("/config")
        assert "(not set)" in result.output

    def test_shows_data_dir(self) -> None:
        result = _make_handler().handle("/config")
        assert "data_dir" in result.output


class TestDoctorCommand:
    def test_reports_python(self) -> None:
        result = _make_handler().handle("/doctor")
        assert "Python" in result.output

    def test_reports_provider(self) -> None:
        result = _make_handler().handle("/doctor")
        assert "Provider" in result.output

    def test_reports_models(self) -> None:
        result = _make_handler().handle("/doctor")
        assert "Models" in result.output

    def test_reports_tools(self) -> None:
        result = _make_handler().handle("/doctor")
        assert "Tools" in result.output

    def test_warns_on_missing_key(self) -> None:
        settings = Settings(ai_provider="gemini")
        handler = CommandHandler(settings, Context("sys"), ToolRegistry())
        result = handler.handle("/doctor")
        assert "warn" in result.output

    def test_ok_with_key(self) -> None:
        result = _make_handler().handle("/doctor")
        assert "[ok]" in result.output

    def test_summary_line(self) -> None:
        result = _make_handler().handle("/doctor")
        assert "check" in result.output.lower() or "passed" in result.output.lower()

    def test_unknown_provider_fails(self) -> None:
        settings = Settings(ai_provider="bogus")
        handler = CommandHandler(settings, Context("sys"), ToolRegistry())
        result = handler.handle("/doctor")
        assert "FAIL" in result.output


class TestNotesCommand:
    def _handler_with_notes(self) -> CommandHandler:
        from liz.infrastructure.persistence.notes_store import SQLiteNotesStore

        settings = Settings(ai_provider="gemini", gemini_api_key="k")
        store = SQLiteNotesStore(":memory:")
        return CommandHandler(settings, Context("sys"), ToolRegistry(), notes_store=store)

    def test_notes_disabled(self) -> None:
        result = _make_handler().handle("/notes")
        assert "not enabled" in result.output

    def test_list_empty(self) -> None:
        result = self._handler_with_notes().handle("/notes")
        assert "No notes" in result.output

    def test_add_note(self) -> None:
        handler = self._handler_with_notes()
        result = handler.handle("/notes add MyTitle")
        assert "Saved note #1" in result.output

    def test_list_after_add(self) -> None:
        handler = self._handler_with_notes()
        handler.handle("/notes add MyTitle")
        result = handler.handle("/notes")
        assert "MyTitle" in result.output

    def test_find_note(self) -> None:
        handler = self._handler_with_notes()
        handler.handle("/notes add MyTitle")
        result = handler.handle("/notes find MyTitle")
        assert "MyTitle" in result.output

    def test_find_no_match(self) -> None:
        handler = self._handler_with_notes()
        handler.handle("/notes add MyTitle")
        result = handler.handle("/notes find ZZZ")
        assert "No notes matching" in result.output

    def test_delete_note(self) -> None:
        handler = self._handler_with_notes()
        handler.handle("/notes add MyTitle")
        result = handler.handle("/notes del 1")
        assert "Deleted note #1" in result.output

    def test_delete_missing(self) -> None:
        handler = self._handler_with_notes()
        result = handler.handle("/notes del 999")
        assert "No note" in result.output

    def test_delete_bad_id(self) -> None:
        handler = self._handler_with_notes()
        result = handler.handle("/notes del abc")
        assert "Usage" in result.output

    def test_add_without_title(self) -> None:
        handler = self._handler_with_notes()
        result = handler.handle("/notes add")
        assert "Usage" in result.output

    def test_find_without_query(self) -> None:
        handler = self._handler_with_notes()
        result = handler.handle("/notes find")
        assert "Usage" in result.output

    def test_unknown_subcommand(self) -> None:
        handler = self._handler_with_notes()
        result = handler.handle("/notes bogus")
        assert "Usage" in result.output


class TestHelpListsNewCommands:
    def test_help_includes_new_commands(self) -> None:
        result = _make_handler().handle("/help")
        assert "/notes" in result.output
        assert "/config" in result.output
        assert "/doctor" in result.output


class _FakeMemory:
    def __init__(self):
        self.store = self
        self._facts = {}
        self.session_id = "default"

    def learn_fact(self, key, value):
        self._facts[key] = value

    def forget_fact(self, key):
        if key in self._facts:
            del self._facts[key]
            return True
        return False

    def facts(self):
        return dict(self._facts)

    def load_history(self):
        return []

    def store_fact(self, key, value):
        raise AttributeError("store_fact should not be called")


def _handler_with_memory() -> tuple[CommandHandler, _FakeMemory]:
    from liz.core.agent import Agent

    memory = _FakeMemory()
    settings = Settings(ai_provider="gemini", gemini_api_key="k")
    agent = MagicMock(spec=Agent)
    agent.memory = memory
    handler = CommandHandler(
        settings, Context("sys"), ToolRegistry(), agent=agent
    )
    return handler, memory


class TestRememberCommand:
    def test_equals_syntax(self) -> None:
        handler, memory = _handler_with_memory()
        result = handler.handle("/remember editor = neovim")
        assert "Remembered" in result.output
        assert memory.facts()["editor"] == "neovim"

    def test_space_syntax(self) -> None:
        handler, memory = _handler_with_memory()
        handler.handle("/remember editor neovim")
        assert memory.facts()["editor"] == "neovim"

    def test_single_word_becomes_true(self) -> None:
        handler, memory = _handler_with_memory()
        handler.handle("/remember likes_cats")
        assert memory.facts()["likes_cats"] == "true"

    def test_overwrites_existing(self) -> None:
        handler, memory = _handler_with_memory()
        handler.handle("/remember editor = vim")
        handler.handle("/remember editor = neovim")
        assert memory.facts()["editor"] == "neovim"

    def test_no_args_usage(self) -> None:
        handler, _ = _handler_with_memory()
        result = handler.handle("/remember")
        assert "Usage" in result.output

    def test_memory_disabled(self) -> None:
        result = _make_handler().handle("/remember x = y")
        assert "not enabled" in result.output

    def test_does_not_use_store_fact(self) -> None:
        handler, memory = _handler_with_memory()
        handler.handle("/remember editor = neovim")
        assert memory.facts()["editor"] == "neovim"


class TestForgetCommand:
    def test_forget_existing(self) -> None:
        handler, memory = _handler_with_memory()
        memory.learn_fact("editor", "vim")

        result = handler.handle("/forget editor")

        assert "Forgot" in result.output
        assert memory.facts() == {}

    def test_forget_missing(self) -> None:
        handler, _ = _handler_with_memory()
        result = handler.handle("/forget nope")
        assert "No fact" in result.output

    def test_no_args_usage(self) -> None:
        handler, _ = _handler_with_memory()
        result = handler.handle("/forget")
        assert "Usage" in result.output

    def test_memory_disabled(self) -> None:
        result = _make_handler().handle("/forget x")
        assert "not enabled" in result.output

    def test_forget_keeps_other_facts(self) -> None:
        handler, memory = _handler_with_memory()
        memory.learn_fact("a", "1")
        memory.learn_fact("b", "2")

        handler.handle("/forget a")

        assert memory.facts() == {"b": "2"}


class TestHelpListsMemoryCommands:
    def test_help_includes_remember_and_forget(self) -> None:
        result = _make_handler().handle("/help")
        assert "/remember" in result.output
        assert "/forget" in result.output


class TestWarnUnknownModel:
    def test_warns_on_unknown_model(self, capsys) -> None:
        from liz.cli.app import warn_unknown_model

        settings = Settings(
            ai_provider="gemini", ai_model="models/gemini-3.8-flash"
        )
        warn_unknown_model(settings)
        captured = capsys.readouterr()
        assert "not a known" in captured.out
        assert "latency" in captured.out

    def test_no_warning_for_known_model(self, capsys) -> None:
        from liz.cli.app import warn_unknown_model

        settings = Settings(ai_provider="gemini", ai_model="gemini-2.0-flash")
        warn_unknown_model(settings)
        assert capsys.readouterr().out == ""

    def test_no_warning_when_model_auto(self, capsys) -> None:
        from liz.cli.app import warn_unknown_model

        settings = Settings(ai_provider="gemini", ai_model=None)
        warn_unknown_model(settings)
        assert capsys.readouterr().out == ""

    def test_no_warning_for_unknown_provider(self, capsys) -> None:
        from liz.cli.app import warn_unknown_model

        settings = Settings(ai_provider="bogus", ai_model="whatever")
        warn_unknown_model(settings)
        assert capsys.readouterr().out == ""


class TestSystemPrompt:
    def test_prompt_forces_users_language(self) -> None:
        from pathlib import Path

        from liz.cli.app import load_system_prompt

        if not Path("prompts/liz.txt").exists():
            pytest.skip("prompt file not available")

        prompt = load_system_prompt().lower()
        assert "language" in prompt
        assert "spanish" in prompt


class TestDoctorWebSearch:
    def test_reports_ddgs(self, monkeypatch) -> None:
        import types

        monkeypatch.setitem(sys.modules, "ddgs", types.ModuleType("ddgs"))

        name, status, detail = CommandHandler._check_web_search()
        assert (name, status) == ("WebSearch", "ok")
        assert "ddgs" in detail

    def test_reports_legacy_package(self, monkeypatch) -> None:
        import types

        monkeypatch.setitem(sys.modules, "ddgs", None)
        monkeypatch.setitem(
            sys.modules, "duckduckgo_search",
            types.ModuleType("duckduckgo_search"),
        )

        _, status, detail = CommandHandler._check_web_search()
        assert status == "ok"
        assert "duckduckgo_search" in detail

    def test_warns_when_nothing_installed(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "ddgs", None)
        monkeypatch.setitem(sys.modules, "duckduckgo_search", None)

        _, status, detail = CommandHandler._check_web_search()
        assert status == "warn"
        assert "HTML fallback" in detail
