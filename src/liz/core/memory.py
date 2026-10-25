"""Conversation memory domain.

`MemoryStore` is the port that persistence implementations must satisfy.
`ConversationMemory` owns the policy of what gets stored and restored so
the Agent never talks to a database directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

from liz.core.models import Message, Role


@dataclass(frozen=True)
class MemoryHit:
    """A search result: one stored message."""

    session_id: str
    role: str
    content: str
    created_at: str


class MemoryStore(ABC):
    """Port for persisting conversation state."""

    @abstractmethod
    def save_message(self, session_id: str, role: str, content: str) -> None:
        ...

    @abstractmethod
    def get_history(
        self, session_id: str, limit: int | None = 50
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
    def delete_fact(self, key: str) -> bool:
        ...

    @abstractmethod
    def list_sessions(self) -> list[str]:
        ...

    @abstractmethod
    def search_messages(
        self, query: str, session_id: str | None = None, limit: int = 20
    ) -> list[MemoryHit]:
        ...

    @abstractmethod
    def list_messages(
        self, session_id: str, limit: int | None = None
    ) -> list[MemoryHit]:
        ...

    @abstractmethod
    def search_facts(self, query: str) -> dict[str, str]:
        ...

    @abstractmethod
    def get_meta(self, key: str) -> str | None:
        ...

    @abstractmethod
    def set_meta(self, key: str, value: str) -> None:
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
        self.history_limit = history_limit
        self.session_id = self.store.get_meta("current_session") or session_id

    def load_history(self) -> list[Message]:
        return self.store.get_history(self.session_id, self.history_limit)

    def remember(self, message: Message) -> None:
        self.store.save_message(
            self.session_id, message.role.value, message.content
        )

    def clear(self) -> None:
        self.store.clear_session(self.session_id)

    # --- facts ---

    def facts(self) -> dict[str, str]:
        return self.store.get_facts()

    def learn_fact(self, key: str, value: str) -> None:
        self.store.save_fact(key, value)

    def forget_fact(self, key: str) -> bool:
        return self.store.delete_fact(key)

    # --- sessions ---

    def sessions(self) -> list[str]:
        found = self.store.list_sessions()
        if self.session_id not in found:
            found.append(self.session_id)
        return sorted(set(found))

    def switch(self, session_id: str) -> str:
        self.session_id = session_id
        self.store.set_meta("current_session", session_id)
        return self.session_id

    # --- search ---

    def recall(
        self, query: str, all_sessions: bool = False
    ) -> tuple[dict[str, str], list[MemoryHit]]:
        """Return matching facts and messages for a query."""
        facts = self.store.search_facts(query)
        session = None if all_sessions else self.session_id
        messages = self.store.search_messages(query, session_id=session)
        return facts, messages

    # --- export / import ---

    def export_session(self, session_id: str | None = None) -> dict:
        """Serialize a session and the facts into a portable dict."""
        target = session_id or self.session_id
        return {
            "version": 1,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "session": target,
            "messages": [
                {
                    "role": hit.role,
                    "content": hit.content,
                    "created_at": hit.created_at,
                }
                for hit in self.store.list_messages(target)
            ],
            "facts": self.store.get_facts(),
        }

    def import_session(self, data: dict, session_id: str | None = None) -> int:
        """Restore an exported payload. Returns messages written."""
        if not isinstance(data, dict) or "messages" not in data:
            raise ValueError("Not a Liz export file")

        messages = data.get("messages") or []
        if not isinstance(messages, list):
            raise ValueError("Export file has no message list")

        target = session_id or data.get("session") or self.session_id
        self.switch(str(target))

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            try:
                Role(role)
            except ValueError:
                continue
            self.store.save_message(target, role, content)

        for key, value in (data.get("facts") or {}).items():
            self.store.save_fact(str(key), str(value))

        return len(messages)

    def close(self) -> None:
        self.store.close()
