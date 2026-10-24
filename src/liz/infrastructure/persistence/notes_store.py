"""SQLite-backed notes store.

Shares the standard library only, like the memory store, so notes work
with zero extra dependencies.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from liz.core.notes import Note, NotesStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_notes_title ON notes(title);
"""


class SQLiteNotesStore(NotesStore):
    def __init__(self, db_path: str | Path) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(str(path))
        self._connection.row_factory = sqlite3.Row
        self._connection.executescript(_SCHEMA)
        self._connection.commit()

    def add(self, title: str, content: str) -> Note:
        cursor = self._connection.execute(
            "INSERT INTO notes (title, content) VALUES (?, ?)",
            (title, content),
        )
        self._connection.commit()

        row = self._connection.execute(
            "SELECT * FROM notes WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()

        return Note(
            id=row["id"],
            title=row["title"],
            content=row["content"],
            created_at=row["created_at"],
        )

    def list(self, limit: int = 20) -> list[Note]:
        rows = self._connection.execute(
            "SELECT * FROM notes ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [self._to_note(row) for row in rows]

    def search(self, query: str) -> list[Note]:
        pattern = f"%{query}%"
        rows = self._connection.execute(
            "SELECT * FROM notes "
            "WHERE title LIKE ? OR content LIKE ? "
            "ORDER BY id DESC",
            (pattern, pattern),
        ).fetchall()
        return [self._to_note(row) for row in rows]

    def delete(self, note_id: int) -> bool:
        cursor = self._connection.execute(
            "DELETE FROM notes WHERE id = ?", (note_id,)
        )
        self._connection.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        self._connection.close()

    @staticmethod
    def _to_note(row: sqlite3.Row) -> Note:
        return Note(
            id=row["id"],
            title=row["title"],
            content=row["content"],
            created_at=row["created_at"],
        )
