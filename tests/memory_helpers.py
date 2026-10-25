"""Shared in-memory `MemoryStore` for tests.

Mirrors the SQLite implementation closely enough that tests exercise
the real `ConversationMemory` policy instead of a stub.
"""

from __future__ import annotations

from liz.core.memory import MemoryHit, MemoryStore
from liz.core.models import Message, Role


class InMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self.messages: list[tuple[str, str, str, str]] = []
        self.facts: dict[str, str] = {}
        self.meta: dict[str, str] = {}
        self.closed = False

    def save_message(self, session_id, role, content):
        self.messages.append((session_id, role, content, "2026-01-01 00:00:00"))

    def get_history(self, session_id, limit=50):
        rows = [m for m in self.messages if m[0] == session_id]
        if limit is not None:
            rows = rows[-limit:]
        return [Message(role=Role(r), content=c) for _, r, c, _ in rows]

    def clear_session(self, session_id):
        self.messages = [m for m in self.messages if m[0] != session_id]

    def save_fact(self, key, value):
        self.facts[key] = value

    def get_facts(self):
        return dict(self.facts)

    def delete_fact(self, key):
        if key in self.facts:
            del self.facts[key]
            return True
        return False

    def list_sessions(self):
        return sorted({m[0] for m in self.messages})

    def search_messages(self, query, session_id=None, limit=20):
        needle = query.lower()
        hits = [
            MemoryHit(session_id=s, role=r, content=c, created_at=t)
            for s, r, c, t in self.messages
            if needle in c.lower()
            and (session_id is None or s == session_id)
        ]
        return hits[:limit]

    def search_facts(self, query):
        needle = query.lower()
        return {
            k: v
            for k, v in self.facts.items()
            if needle in k.lower() or needle in v.lower()
        }

    def list_messages(self, session_id, limit=None):
        rows = [m for m in self.messages if m[0] == session_id]
        if limit is not None:
            rows = rows[-limit:]
        return [
            MemoryHit(session_id=s, role=r, content=c, created_at=t)
            for s, r, c, t in rows
        ]

    def get_meta(self, key):
        return self.meta.get(key)

    def set_meta(self, key, value):
        self.meta[key] = value

    def close(self):
        self.closed = True
