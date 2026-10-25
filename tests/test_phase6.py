import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from liz.config.settings import Settings
from liz.core.agent import Agent
from liz.core.context import Context
from liz.core.memory import ConversationMemory, MemoryHit
from liz.core.models import AIResponse, Message, Role, ToolCall
from liz.core.summarizer import ContextSummarizer
from liz.infrastructure.ai.base import AIProvider
from liz.infrastructure.persistence.memory_store import SQLiteMemoryStore
from tests.memory_helpers import InMemoryStore


def _memory(store=None, session="default") -> ConversationMemory:
    return ConversationMemory(store or InMemoryStore(), session_id=session)


class TestSessions:
    def test_lists_sessions(self) -> None:
        memory = _memory()
        memory.switch("work")
        memory.remember(Message(role=Role.USER, content="hola"))
        memory.switch("personal")
        memory.remember(Message(role=Role.USER, content="otro"))

        assert memory.sessions() == ["personal", "work"]

    def test_current_session_included(self) -> None:
        memory = _memory(session="fresh")
        assert "fresh" in memory.sessions()

    def test_switch_persists_to_meta(self) -> None:
        store = InMemoryStore()
        memory = _memory(store)
        memory.switch("work")

        assert store.get_meta("current_session") == "work"

    def test_session_restored_from_meta(self) -> None:
        store = InMemoryStore()
        ConversationMemory(store).switch("work")

        reopened = ConversationMemory(store)
        assert reopened.session_id == "work"

    def test_switch_returns_name(self) -> None:
        memory = _memory()
        assert memory.switch("x") == "x"

    def test_sessions_isolated(self) -> None:
        memory = _memory()
        memory.switch("a")
        memory.remember(Message(role=Role.USER, content="para a"))
        memory.switch("b")
        memory.remember(Message(role=Role.USER, content="para b"))

        memory.switch("a")
        assert len(memory.load_history()) == 1
        assert memory.load_history()[0].content == "para a"


class TestAgentSwitchSession:
    def _agent(self, memory) -> Agent:
        settings = Settings(ai_provider="gemini", gemini_api_key="k")
        with patch("liz.core.agent.create_provider") as mock:
            mock.return_value = MagicMock(spec=AIProvider)
            return Agent(settings, "sys", memory=memory)

    def test_switch_reloads_context(self) -> None:
        store = InMemoryStore()
        memory = ConversationMemory(store, session_id="a")
        memory.remember(Message(role=Role.USER, content="old in a"))

        agent = self._agent(memory)
        assert len(agent.context.get_messages()) > 1

        memory.switch("b")
        agent.switch_session("b")
        contents = [m.content for m in agent.context.get_messages()]
        assert "old in a" not in contents

    def test_switch_without_memory_is_noop(self) -> None:
        settings = Settings(ai_provider="gemini", gemini_api_key="k")
        with patch("liz.core.agent.create_provider") as mock:
            mock.return_value = MagicMock(spec=AIProvider)
            agent = Agent(settings, "sys")

        assert agent.switch_session("anything") == "anything"


class TestRecall:
    def test_finds_facts(self) -> None:
        memory = _memory()
        memory.learn_fact("editor", "neovim")

        facts, messages = memory.recall("editor")
        assert facts == {"editor": "neovim"}

    def test_finds_messages(self) -> None:
        memory = _memory()
        memory.remember(Message(role=Role.USER, content="me gusta Hermosillo"))

        _, messages = memory.recall("hermosillo")
        assert len(messages) == 1
        assert "Hermosillo" in messages[0].content

    def test_search_is_case_insensitive(self) -> None:
        memory = _memory()
        memory.remember(Message(role=Role.USER, content="NEOVIM aqui"))

        _, messages = memory.recall("neovim")
        assert len(messages) == 1

    def test_no_match(self) -> None:
        memory = _memory()
        memory.learn_fact("a", "b")

        facts, messages = memory.recall("zzzz")
        assert facts == {}
        assert messages == []

    def test_all_sessions_option(self) -> None:
        memory = _memory()
        memory.switch("other")
        memory.remember(Message(role=Role.USER, content="secreto trabajo"))

        memory.switch("home")
        _, scoped = memory.recall("secreto")
        assert scoped == []

        _, everything = memory.recall("secreto", all_sessions=True)
        assert len(everything) == 1

    def test_matches_fact_values(self) -> None:
        memory = _memory()
        memory.learn_fact("city", "Hermosillo")

        facts, _ = memory.recall("hermos")
        assert facts == {"city": "Hermosillo"}


class TestSQLiteRecall:
    def test_search_messages(self, tmp_path) -> None:
        store = SQLiteMemoryStore(tmp_path / "m.db")
        store.save_message("s1", "user", "¿dónde está mi llave?")
        store.save_message("s1", "assistant", "no sé")

        hits = store.search_messages("llave")
        assert len(hits) == 1
        assert hits[0].session_id == "s1"
        store.close()

    def test_search_scoped_to_session(self, tmp_path) -> None:
        store = SQLiteMemoryStore(tmp_path / "m.db")
        store.save_message("a", "user", "hola")
        store.save_message("b", "user", "hola")

        assert len(store.search_messages("hola", session_id="a")) == 1
        store.close()

    def test_list_sessions(self, tmp_path) -> None:
        store = SQLiteMemoryStore(tmp_path / "m.db")
        store.save_message("x", "user", "1")
        store.save_message("y", "user", "2")

        assert store.list_sessions() == ["x", "y"]
        store.close()

    def test_meta_round_trip(self, tmp_path) -> None:
        store = SQLiteMemoryStore(tmp_path / "m.db")
        assert store.get_meta("missing") is None

        store.set_meta("current_session", "work")
        store.set_meta("current_session", "home")

        assert store.get_meta("current_session") == "home"
        store.close()

    def test_list_messages_all(self, tmp_path) -> None:
        store = SQLiteMemoryStore(tmp_path / "m.db")
        store.save_message("s", "user", "1")
        store.save_message("s", "assistant", "2")

        rows = store.list_messages("s")
        assert [r.content for r in rows] == ["1", "2"]
        store.close()

    def test_search_facts(self, tmp_path) -> None:
        store = SQLiteMemoryStore(tmp_path / "m.db")
        store.save_fact("editor", "neovim")

        assert store.search_facts("neov") == {"editor": "neovim"}
        assert store.search_facts("zzz") == {}
        store.close()


class TestExportImport:
    def test_export_shape(self) -> None:
        memory = _memory()
        memory.learn_fact("a", "1")
        memory.remember(Message(role=Role.USER, content="hola"))

        data = memory.export_session()

        assert data["version"] == 1
        assert data["session"] == "default"
        assert len(data["messages"]) == 1
        assert data["facts"] == {"a": "1"}
        assert "exported_at" in data

    def test_export_specific_session(self) -> None:
        memory = _memory()
        memory.switch("work")
        memory.remember(Message(role=Role.USER, content="trabajo"))

        data = memory.export_session("work")
        assert data["session"] == "work"
        assert len(data["messages"]) == 1

    def test_round_trip(self) -> None:
        source = _memory()
        source.learn_fact("city", "Hermosillo")
        source.remember(Message(role=Role.USER, content="primero"))
        source.remember(Message(role=Role.ASSISTANT, content="segundo"))
        payload = source.export_session()

        target = _memory()
        written = target.import_session(payload, session_id="restored")

        assert written == 2
        assert target.session_id == "restored"
        history = target.load_history()
        assert [m.content for m in history] == ["primero", "segundo"]
        assert target.facts() == {"city": "Hermosillo"}

    def test_import_rejects_garbage(self) -> None:
        memory = _memory()
        with pytest.raises(ValueError, match="Not a Liz export"):
            memory.import_session({"nope": True})

    def test_import_rejects_non_list_messages(self) -> None:
        memory = _memory()
        with pytest.raises(ValueError, match="no message list"):
            memory.import_session({"messages": "not-a-list"})

    def test_import_skips_bad_roles(self) -> None:
        memory = _memory()
        data = {
            "messages": [
                {"role": "user", "content": "ok"},
                {"role": "banana", "content": "skip"},
            ]
        }

        memory.import_session(data, session_id="s")

        history = memory.load_history()
        assert [m.content for m in history] == ["ok"]

    def test_export_to_file(self, tmp_path) -> None:
        from liz.cli.commands import CommandHandler
        from liz.tools.registry import ToolRegistry

        memory = _memory()
        memory.remember(Message(role=Role.USER, content="hola"))
        agent = MagicMock()
        agent.memory = memory

        settings = Settings(ai_provider="gemini", gemini_api_key="k")
        handler = CommandHandler(
            settings, Context("sys"), ToolRegistry(), agent=agent
        )

        target = tmp_path / "out.json"
        result = handler.handle(f"/export {target}")

        assert "Exported" in result.output
        assert target.exists()
        assert json.loads(target.read_text())["session"] == "default"


class _FakeProvider(AIProvider):
    def __init__(self, content="short summary") -> None:
        self.model = "fake"
        self.content = content
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        return AIResponse(content=self.content, model=self.model)


class TestSummarizer:
    def _context(self, n: int) -> Context:
        context = Context("sys", max_messages=500)
        for index in range(n):
            role = Role.USER if index % 2 == 0 else Role.ASSISTANT
            context.messages.append(
                Message(role=role, content=f"turn {index}")
            )
        return context

    def test_not_summarized_below_threshold(self) -> None:
        summarizer = ContextSummarizer(_FakeProvider(), threshold=30)
        context = self._context(10)

        assert summarizer.should_summarize(context) is False

    def test_summarized_above_threshold(self) -> None:
        summarizer = ContextSummarizer(_FakeProvider(), threshold=30)
        context = self._context(40)

        assert summarizer.should_summarize(context) is True

    def test_summarize_shrinks_context(self) -> None:
        provider = _FakeProvider("resumen")
        summarizer = ContextSummarizer(
            provider, threshold=30, keep_recent=10
        )
        context = self._context(40)
        before = len(context.messages)

        changed = summarizer.summarize(context)

        assert changed is True
        assert len(context.messages) < before
        assert len(context.messages) == 12  # system + summary + 10 recent

    def test_summary_content_present(self) -> None:
        summarizer = ContextSummarizer(
            _FakeProvider("this is the summary"), threshold=10, keep_recent=4
        )
        context = self._context(20)

        summarizer.summarize(context)

        assert context.messages[1].content.endswith("this is the summary")

    def test_system_prompt_kept_first(self) -> None:
        summarizer = ContextSummarizer(
            _FakeProvider(), threshold=10, keep_recent=4
        )
        context = self._context(20)

        summarizer.summarize(context)

        assert context.messages[0].role is Role.SYSTEM
        assert context.messages[0].content == "sys"

    def test_recent_messages_survive(self) -> None:
        summarizer = ContextSummarizer(
            _FakeProvider(), threshold=10, keep_recent=4
        )
        context = self._context(20)

        summarizer.summarize(context)

        contents = [m.content for m in context.messages]
        assert contents[-1] == "turn 19"

    def test_noop_when_too_small(self) -> None:
        summarizer = ContextSummarizer(_FakeProvider(), threshold=100)
        context = self._context(3)

        assert summarizer.summarize(context) is False

    def test_provider_failure_falls_back(self) -> None:
        provider = _FakeProvider()
        provider.chat = MagicMock(side_effect=RuntimeError("down"))
        summarizer = ContextSummarizer(
            provider, threshold=10, keep_recent=4
        )
        context = self._context(20)

        changed = summarizer.summarize(context)

        assert changed is True
        assert "summary unavailable" in context.messages[1].content

    def test_keeps_tool_pairing_intact(self) -> None:
        summarizer = ContextSummarizer(
            _FakeProvider(), threshold=5, keep_recent=6
        )
        context = Context("sys", max_messages=500)
        for index in range(9):
            context.messages.append(
                Message(role=Role.USER, content=f"q{index}")
            )
        call = ToolCall(id="c1", name="x", arguments={})
        context.messages.append(
            Message(role=Role.ASSISTANT, content="", tool_calls=[call])
        )
        context.messages.append(
            Message(role=Role.TOOL, content="r", tool_call_id="c1")
        )

        changed = summarizer.summarize(context)

        assert changed is True
        roles = [m.role for m in context.messages]
        # the tool result must still be preceded by its call
        tool_index = roles.index(Role.TOOL)
        assert context.messages[tool_index - 1].tool_calls

    def test_collapses_tool_pair_together(self) -> None:
        """An older call/result pair must never be split apart."""
        summarizer = ContextSummarizer(
            _FakeProvider(), threshold=5, keep_recent=4
        )
        context = Context("sys", max_messages=500)
        for index in range(4):
            context.messages.append(
                Message(role=Role.USER, content=f"q{index}")
            )
        call = ToolCall(id="c1", name="x", arguments={})
        context.messages.append(
            Message(role=Role.ASSISTANT, content="", tool_calls=[call])
        )
        context.messages.append(
            Message(role=Role.TOOL, content="r", tool_call_id="c1")
        )
        for index in range(6):
            context.messages.append(
                Message(role=Role.USER, content=f"tail{index}")
            )

        summarizer.summarize(context)

        messages = context.messages
        for index, message in enumerate(messages):
            if message.role is Role.TOOL:
                assert index > 0 and messages[index - 1].tool_calls
        # either the pair survived together or both were collapsed
        tool_count = sum(1 for m in messages if m.role is Role.TOOL)
        call_count = sum(1 for m in messages if m.tool_calls)
        assert tool_count == call_count

    def test_transcript_ignores_tool_messages(self) -> None:
        summarizer = ContextSummarizer(_FakeProvider(), threshold=5, keep_recent=4)
        context = Context("sys", max_messages=500)
        context.messages.append(Message(role=Role.USER, content="visible"))
        context.messages.append(
            Message(role=Role.TOOL, content="internal", tool_call_id="c")
        )
        for index in range(10):
            context.messages.append(
                Message(role=Role.USER, content=f"pad{index}")
            )

        summarizer.summarize(context)

        # summary came from the provider, so transcript must have fed it
        assert summarizer.provider.calls == 1
