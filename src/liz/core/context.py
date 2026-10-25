from liz.core.models import Message, Role, ToolCall


class Context:
    def __init__(self, system_prompt: str, max_messages: int = 50) -> None:
        self.system_prompt = system_prompt
        self.max_messages = max_messages
        self.messages: list[Message] = [
            Message(role=Role.SYSTEM, content=system_prompt)
        ]

    def add_user_message(self, content: str) -> None:
        self.messages.append(Message(role=Role.USER, content=content))
        self._trim()

    def add_assistant_message(
        self,
        content: str,
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        self.messages.append(
            Message(
                role=Role.ASSISTANT,
                content=content,
                tool_calls=tool_calls or [],
            )
        )
        self._trim()

    def add_tool_result(
        self, tool_call_id: str | None, content: str
    ) -> None:
        self.messages.append(
            Message(
                role=Role.TOOL,
                content=content,
                tool_call_id=tool_call_id,
            )
        )
        self._trim()

    def get_messages(self) -> list[Message]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages = [
            Message(role=Role.SYSTEM, content=self.system_prompt)
        ]

    def _trim(self) -> None:
        if len(self.messages) <= self.max_messages:
            return

        trimmed = [
            self.messages[0],
            *self.messages[-(self.max_messages - 1) :],
        ]

        # A tool result whose call was trimmed away is invalid for
        # providers that require the pairing, so drop orphans.
        while len(trimmed) > 1 and trimmed[1].role == Role.TOOL:
            trimmed.pop(1)

        self.messages = trimmed
