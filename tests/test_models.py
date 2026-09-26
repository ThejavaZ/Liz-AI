from liz.core.models import AIResponse, Message, Role


class TestRole:
    def test_role_values(self) -> None:
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"

    def test_role_count(self) -> None:
        assert len(Role) == 3


class TestMessage:
    def test_create_message(self) -> None:
        msg = Message(role=Role.USER, content="Hello")
        assert msg.role == Role.USER
        assert msg.content == "Hello"

    def test_message_equality(self) -> None:
        msg1 = Message(role=Role.USER, content="Hello")
        msg2 = Message(role=Role.USER, content="Hello")
        assert msg1 == msg2

    def test_message_inequality(self) -> None:
        msg1 = Message(role=Role.USER, content="Hello")
        msg2 = Message(role=Role.USER, content="Different")
        assert msg1 != msg2


class TestAIResponse:
    def test_create_response(self) -> None:
        resp = AIResponse(content="Hi", model="gpt-4")
        assert resp.content == "Hi"
        assert resp.model == "gpt-4"
        assert resp.usage == {}

    def test_response_with_usage(self) -> None:
        usage = {"input_tokens": 100, "output_tokens": 50}
        resp = AIResponse(content="Hi", model="gpt-4", usage=usage)
        assert resp.usage == usage
