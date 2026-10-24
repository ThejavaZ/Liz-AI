"""Conversation memory domain.

`MemoryStore` is the port that persistence implementations must satisfy.
`ConversationMemory` owns the policy of what gets stored and restored so
the Agent never talks to a database directly.
"""

from abc import ABC, abstractmethod

from liz.core.models import Message


class MemoryStore(ABC):
    """Port for persisting conversation state."""

    @abstractmethod
    def save_message(self, session_id: str, role: str, content: str) -> None:
        ...

    @abstractmethod
    def get_history(
        self, session_id: str, limit: int = 50
    ) -> list[Message]:
        ...

    @abstractmethod
    def clear_session(self, session_id: str) -> None:
        ...

    @abstractmethod
    def save_fact(self, key: str, value: str) -> None:
        ...

    @abstractmethod
    def get_facts(self) -> dict[str, str]:
        ...

    @abstractmethod
    def close(self) -> None:
        ...


class ConversationMemory:
    """Coordinates what to persist and restore for a single session."""

    def __init__(
        self,
        store: MemoryStore,
        session_id: str = "default",
        history_limit: int = 20,
    ) -> None:
        self.store = store
        self.session_id = session_id
        self.history_limit = history_limit

    def load_history(self) -> list[Message]:
        return self.store.get_history(self.session_id, self.history_limit)

    def remember(self, message: Message) -> None:
        self.store.save_message(
            self.session_id, message.role.value, message.content
        )

    def clear(self) -> None:
        self.store.clear_session(self.session_id)

    def facts(self) -> dict[str, str]:
        return self.store.get_facts()

    def learn_fact(self, key: str, value: str) -> None:
        self.store.save_fact(key, value)

    def close(self) -> None:
        self.store.close()
