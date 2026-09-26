from liz.core.context import Context
from liz.core.models import Role


class TestContext:
    def test_init_with_system_prompt(self) -> None:
        ctx = Context(system_prompt="You are helpful")
        assert ctx.system_prompt == "You are helpful"
        assert ctx.max_messages == 50
        assert len(ctx.messages) == 1
        assert ctx.messages[0].role == Role.SYSTEM
        assert ctx.messages[0].content == "You are helpful"

    def test_add_user_message(self) -> None:
        ctx = Context(system_prompt="sys")
        ctx.add_user_message("Hello")
        assert len(ctx.messages) == 2
        assert ctx.messages[1].role == Role.USER
        assert ctx.messages[1].content == "Hello"

    def test_add_assistant_message(self) -> None:
        ctx = Context(system_prompt="sys")
        ctx.add_assistant_message("Hi there")
        assert len(ctx.messages) == 2
        assert ctx.messages[1].role == Role.ASSISTANT
        assert ctx.messages[1].content == "Hi there"

    def test_conversation_flow(self) -> None:
        ctx = Context(system_prompt="sys")
        ctx.add_user_message("Q1")
        ctx.add_assistant_message("A1")
        ctx.add_user_message("Q2")
        assert len(ctx.messages) == 4

    def test_get_messages_returns_copy(self) -> None:
        ctx = Context(system_prompt="sys")
        msgs = ctx.get_messages()
        ctx.add_user_message("Hello")
        assert len(msgs) == 1
        assert len(ctx.messages) == 2

    def test_clear_resets_to_system_only(self) -> None:
        ctx = Context(system_prompt="sys")
        ctx.add_user_message("Hello")
        ctx.add_assistant_message("Hi")
        ctx.clear()
        assert len(ctx.messages) == 1
        assert ctx.messages[0].role == Role.SYSTEM

    def test_trim_when_exceeding_max(self) -> None:
        ctx = Context(system_prompt="sys", max_messages=4)
        ctx.add_user_message("1")
        ctx.add_assistant_message("2")
        ctx.add_user_message("3")
        ctx.add_assistant_message("4")
        assert len(ctx.messages) == 4
        ctx.add_user_message("5")
        assert len(ctx.messages) == 4
        assert ctx.messages[0].role == Role.SYSTEM
        assert ctx.messages[1].content == "3"
