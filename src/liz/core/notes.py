"""Notes domain.

Notes are short user-authored records that outlive a conversation.
`NotesStore` is the port; persistence implementations live in
infrastructure.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Note:
    id: int
    title: str
    content: str
    created_at: str


class NotesStore(ABC):
    """Port for persisting notes."""

    @abstractmethod
    def add(self, title: str, content: str) -> Note:
        ...

    @abstractmethod
    def list(self, limit: int = 20) -> list[Note]:
        ...

    @abstractmethod
    def search(self, query: str) -> list[Note]:
        ...

    @abstractmethod
    def delete(self, note_id: int) -> bool:
        ...

    @abstractmethod
    def close(self) -> None:
        ...
