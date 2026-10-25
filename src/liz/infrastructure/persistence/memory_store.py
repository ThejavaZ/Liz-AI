"""SQLite-backed memory store.

Uses only the standard library so persistence works without extra
dependencies. The database lives under the configured data directory.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from liz.core.memory import MemoryHit, MemoryStore
from liz.core.models import Message, Role

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, id);

CREATE TABLE IF NOT EXISTS facts (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class SQLiteMemoryStore(MemoryStore):
    def __init__(self, db_path: str | Path) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(str(path))
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    def save_message(self, session_id: str, role: str, content: str) -> None:
        self._connection.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        self._connection.commit()

    def get_history(
        self, session_id: str, limit: int | None = 50
    ) -> list[Message]:
        if limit is None:
            rows = self._connection.execute(
                "SELECT role, content FROM messages "
                "WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT role, content FROM messages "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            rows = list(reversed(rows))

        return [Message(role=Role(row["role"]), content=row["content"]) for row in rows]

    def clear_session(self, session_id: str) -> None:
        self._connection.execute(
            "DELETE FROM messages WHERE session_id = ?", (session_id,)
        )
        self._connection.commit()

    def save_fact(self, key: str, value: str) -> None:
        self._connection.execute(
            "INSERT INTO facts (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
            "updated_at = CURRENT_TIMESTAMP",
            (key, value),
        )
        self._connection.commit()

    def get_facts(self) -> dict[str, str]:
        rows = self._connection.execute(
            "SELECT key, value FROM facts ORDER BY key"
        ).fetchall()
        return {row["key"]: row["value"] for row in rows}

    def delete_fact(self, key: str) -> bool:
        cursor = self._connection.execute(
            "DELETE FROM facts WHERE key = ?", (key,)
        )
        self._connection.commit()
        return cursor.rowcount > 0

    def list_sessions(self) -> list[str]:
        rows = self._connection.execute(
            "SELECT DISTINCT session_id FROM messages ORDER BY session_id"
        ).fetchall()
        return [row["session_id"] for row in rows]

    def search_messages(
        self, query: str, session_id: str | None = None, limit: int = 20
    ) -> list[MemoryHit]:
        pattern = f"%{query}%"

        if session_id is None:
            rows = self._connection.execute(
                "SELECT session_id, role, content, created_at FROM messages "
                "WHERE content LIKE ? ORDER BY id DESC LIMIT ?",
                (pattern, limit),
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT session_id, role, content, created_at FROM messages "
                "WHERE content LIKE ? AND session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (pattern, session_id, limit),
            ).fetchall()

        return [
            MemoryHit(
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def search_facts(self, query: str) -> dict[str, str]:
        pattern = f"%{query}%"
        rows = self._connection.execute(
            "SELECT key, value FROM facts "
            "WHERE key LIKE ? OR value LIKE ? ORDER BY key",
            (pattern, pattern),
        ).fetchall()
        return {row["key"]: row["value"] for row in rows}

    def list_messages(
        self, session_id: str, limit: int | None = None
    ) -> list[MemoryHit]:
        if limit is None:
            rows = self._connection.execute(
                "SELECT session_id, role, content, created_at FROM messages "
                "WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT session_id, role, content, created_at FROM messages "
                "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            rows = list(reversed(rows))

        return [
            MemoryHit(
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def get_meta(self, key: str) -> str | None:
        row = self._connection.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self._connection.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()
