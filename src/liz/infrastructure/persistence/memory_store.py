"""SQLite-backed memory store.

Uses only the standard library so persistence works without extra
dependencies. The database lives under the configured data directory.
"""

import sqlite3
from pathlib import Path

from liz.core.memory import MemoryStore
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
        self, session_id: str, limit: int = 50
    ) -> list[Message]:
        rows = self._connection.execute(
            "SELECT role, content FROM messages "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()

        history = [
            Message(role=Role(row["role"]), content=row["content"])
            for row in reversed(rows)
        ]
        return history

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

    def close(self) -> None:
        self._connection.close()
