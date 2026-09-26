from liz.core.models import Message, Role


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

    def add_assistant_message(self, content: str) -> None:
        self.messages.append(Message(role=Role.ASSISTANT, content=content))
        self._trim()

    def get_messages(self) -> list[Message]:
        return list(self.messages)

    def clear(self) -> None:
        self.messages = [
            Message(role=Role.SYSTEM, content=self.system_prompt)
        ]

    def _trim(self) -> None:
        if len(self.messages) > self.max_messages:
            system = self.messages[0]
            recent = self.messages[-(self.max_messages - 1) :]
            self.messages = [system] + recent
