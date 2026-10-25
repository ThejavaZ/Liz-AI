import pytest
from unittest.mock import MagicMock, patch

from liz.config.settings import Settings
from liz.core.agent import Agent
from liz.core.memory import ConversationMemory
from liz.core.models import AIResponse, Message, Role
from liz.infrastructure.ai.base import AIProvider
from liz.infrastructure.persistence.memory_store import SQLiteMemoryStore
from tests.memory_helpers import InMemoryStore


class TestConversationMemory:
    def test_remember_and_load(self) -> None:
        store = InMemoryStore()
        memory = ConversationMemory(store, session_id="s1")

        memory.remember(Message(role=Role.USER, content="hello"))
        memory.remember(Message(role=Role.ASSISTANT, content="hi"))

        history = memory.load_history()
        assert len(history) == 2
        assert history[0].content == "hello"
        assert history[1].content == "hi"

    def test_session_isolation(self) -> None:
        store = InMemoryStore()
        a = ConversationMemory(store, session_id="a")
        b = ConversationMemory(store, session_id="b")

        a.remember(Message(role=Role.USER, content="for a"))
        b.remember(Message(role=Role.USER, content="for b"))

        assert len(a.load_history()) == 1
        assert len(b.load_history()) == 1

    def test_clear_session(self) -> None:
        store = InMemoryStore()
        memory = ConversationMemory(store, session_id="s1")
        memory.remember(Message(role=Role.USER, content="hello"))

        memory.clear()

        assert memory.load_history() == []

    def test_facts(self) -> None:
        store = InMemoryStore()
        memory = ConversationMemory(store)

        memory.learn_fact("user_name", "Javier")
        memory.learn_fact("editor", "VSCode")

        facts = memory.facts()
        assert facts["user_name"] == "Javier"
        assert facts["editor"] == "VSCode"

    def test_fact_overwrite(self) -> None:
        store = InMemoryStore()
        memory = ConversationMemory(store)

        memory.learn_fact("editor", "Vim")
        memory.learn_fact("editor", "Neovim")

        assert memory.facts()["editor"] == "Neovim"

    def test_close_delegates(self) -> None:
        store = InMemoryStore()
        memory = ConversationMemory(store)
        memory.close()
        assert store.closed is True

    def test_history_limit(self) -> None:
        store = InMemoryStore()
        memory = ConversationMemory(store, session_id="s1", history_limit=3)

        for i in range(10):
            memory.remember(Message(role=Role.USER, content=f"msg{i}"))

        history = memory.load_history()
        assert len(history) == 3
        assert history[-1].content == "msg9"


class TestSQLiteMemoryStore:
    def _store(self, tmp_path) -> SQLiteMemoryStore:
        return SQLiteMemoryStore(tmp_path / "test.db")

    def test_persists_messages(self, tmp_path) -> None:
        store = self._store(tmp_path)
        store.save_message("s1", "user", "hello")
        store.save_message("s1", "assistant", "hi")
        store.close()

        reopened = self._store(tmp_path)
        history = reopened.get_history("s1")
        assert len(history) == 2
        assert history[0].content == "hello"
        reopened.close()

    def test_clear_session(self, tmp_path) -> None:
        store = self._store(tmp_path)
        store.save_message("s1", "user", "hello")
        store.clear_session("s1")
        assert store.get_history("s1") == []
        store.close()

    def test_facts_persist(self, tmp_path) -> None:
        store = self._store(tmp_path)
        store.save_fact("name", "Javier")
        store.close()

        reopened = self._store(tmp_path)
        assert reopened.get_facts() == {"name": "Javier"}
        reopened.close()

    def test_fact_upsert(self, tmp_path) -> None:
        store = self._store(tmp_path)
        store.save_fact("name", "a")
        store.save_fact("name", "b")
        assert store.get_facts() == {"name": "b"}
        store.close()

    def test_creates_parent_dir(self, tmp_path) -> None:
        target = tmp_path / "nested" / "deeper" / "liz.db"
        store = SQLiteMemoryStore(target)
        assert target.exists()
        store.close()

    def test_history_order(self, tmp_path) -> None:
        store = self._store(tmp_path)
        store.save_message("s1", "user", "first")
        store.save_message("s1", "assistant", "second")
        store.save_message("s1", "user", "third")

        history = store.get_history("s1")
        assert [m.content for m in history] == ["first", "second", "third"]
        store.close()


class _FakeProvider(AIProvider):
    def __init__(self, content: str = "hello") -> None:
        self.model = "test-model"
        self.content = content

    def chat(self, messages, tools=None):
        return AIResponse(content=self.content, model=self.model)


class TestAgentWithMemory:
    def _agent(self, memory, content="hello") -> Agent:
        settings = Settings(ai_provider="gemini", gemini_api_key="k")
        with patch("liz.core.agent.create_provider") as mock:
            mock.return_value = _FakeProvider(content)
            return Agent(settings, "sys", memory=memory)

    def test_saves_user_and_assistant(self) -> None:
        store = InMemoryStore()
        memory = ConversationMemory(store, session_id="s1")
        agent = self._agent(memory)

        agent.chat("hi there")

        history = store.get_history("s1")
        assert len(history) == 2
        assert history[0].role == Role.USER
        assert history[1].role == Role.ASSISTANT

    def test_restores_history_on_init(self) -> None:
        store = InMemoryStore()
        memory = ConversationMemory(store, session_id="s1")
        memory.remember(Message(role=Role.USER, content="old question"))
        memory.remember(Message(role=Role.ASSISTANT, content="old answer"))

        agent = self._agent(memory)

        roles = [m.role for m in agent.context.get_messages()]
        assert Role.USER in roles
        contents = [m.content for m in agent.context.get_messages()]
        assert "old question" in contents

    def test_clear_conversation_clears_memory(self) -> None:
        store = InMemoryStore()
        memory = ConversationMemory(store, session_id="s1")
        agent = self._agent(memory)
        agent.chat("hi")

        agent.clear_conversation()

        assert store.get_history("s1") == []
        assert len(agent.context.get_messages()) == 1

    def test_no_memory_is_fine(self) -> None:
        settings = Settings(ai_provider="gemini", gemini_api_key="k")
        with patch("liz.core.agent.create_provider") as mock:
            mock.return_value = _FakeProvider()
            agent = Agent(settings, "sys")

        assert agent.memory is None
        agent.chat("hi")
        assert len(agent.context.get_messages()) == 3


class TestForgetFact:
    def test_forget_existing(self) -> None:
        store = InMemoryStore()
        memory = ConversationMemory(store)
        memory.learn_fact("editor", "neovim")

        assert memory.forget_fact("editor") is True
        assert "editor" not in memory.facts()

    def test_forget_missing(self) -> None:
        store = InMemoryStore()
        memory = ConversationMemory(store)

        assert memory.forget_fact("nope") is False

    def test_forget_keeps_others(self) -> None:
        store = InMemoryStore()
        memory = ConversationMemory(store)
        memory.learn_fact("a", "1")
        memory.learn_fact("b", "2")

        memory.forget_fact("a")

        assert memory.facts() == {"b": "2"}


class TestSQLiteForgetFact:
    def test_forget_existing(self, tmp_path) -> None:
        store = SQLiteMemoryStore(tmp_path / "m.db")
        store.save_fact("editor", "neovim")

        assert store.delete_fact("editor") is True
        assert store.get_facts() == {}
        store.close()

    def test_forget_missing(self, tmp_path) -> None:
        store = SQLiteMemoryStore(tmp_path / "m.db")
        assert store.delete_fact("nope") is False
        store.close()

    def test_forget_persists(self, tmp_path) -> None:
        SQLiteMemoryStore(tmp_path / "m.db").save_fact("a", "1")
        store = SQLiteMemoryStore(tmp_path / "m.db")
        store.delete_fact("a")
        store.close()

        reopened = SQLiteMemoryStore(tmp_path / "m.db")
        assert reopened.get_facts() == {}
        reopened.close()
