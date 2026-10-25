import pytest

from liz.core.memory import ConversationMemory
from liz.tools.memory import Memory
from liz.tools.models import ToolStatus
from tests.memory_helpers import InMemoryStore


def _memory() -> tuple[Memory, InMemoryStore]:
    store = InMemoryStore()
    return Memory(ConversationMemory(store)), store


class TestMemoryTool:
    def test_name(self) -> None:
        tool, _ = _memory()
        assert tool.name == "memory"

    def test_description(self) -> None:
        tool, _ = _memory()
        assert "remember" in tool.description.lower()

    def test_parameters_schema(self) -> None:
        schema = Memory(ConversationMemory(InMemoryStore())).parameters_schema
        assert "action" in schema["properties"]
        assert "action" in schema["required"]
        assert "remember" in schema["properties"]["action"]["enum"]

    def test_remember(self) -> None:
        tool, store = _memory()
        result = tool.execute(action="remember", key="editor", value="neovim")

        assert result.status == ToolStatus.SUCCESS
        assert store.facts == {"editor": "neovim"}

    def test_remember_requires_key(self) -> None:
        tool, _ = _memory()
        result = tool.execute(action="remember", key="", value="x")
        assert result.status == ToolStatus.ERROR

    def test_remember_requires_value(self) -> None:
        tool, _ = _memory()
        result = tool.execute(action="remember", key="k", value="")
        assert result.status == ToolStatus.ERROR

    def test_remember_overwrites(self) -> None:
        tool, store = _memory()
        tool.execute(action="remember", key="editor", value="vim")
        tool.execute(action="remember", key="editor", value="neovim")

        assert store.facts["editor"] == "neovim"

    def test_forget(self) -> None:
        tool, store = _memory()
        store.save_fact("editor", "vim")

        result = tool.execute(action="forget", key="editor")

        assert "Forgot" in result.data
        assert store.facts == {}

    def test_forget_missing(self) -> None:
        tool, _ = _memory()
        result = tool.execute(action="forget", key="nope")
        assert result.status == ToolStatus.ERROR

    def test_forget_requires_key(self) -> None:
        tool, _ = _memory()
        result = tool.execute(action="forget", key="")
        assert result.status == ToolStatus.ERROR

    def test_list_empty(self) -> None:
        tool, _ = _memory()
        result = tool.execute(action="list")
        assert "No facts" in result.data

    def test_list_facts(self) -> None:
        tool, store = _memory()
        store.save_fact("a", "1")
        store.save_fact("b", "2")

        result = tool.execute(action="list")
        assert "a = 1" in result.data
        assert "b = 2" in result.data

    def test_unknown_action(self) -> None:
        tool, _ = _memory()
        result = tool.execute(action="bogus")
        assert result.status == ToolStatus.ERROR
        assert "Unknown action" in result.error

    def test_store_error_reported(self) -> None:
        tool, _ = _memory()

        def boom(key, value):
            raise RuntimeError("db down")

        tool.memory.store.save_fact = boom

        result = tool.execute(action="remember", key="k", value="v")
        assert result.status == ToolStatus.ERROR
