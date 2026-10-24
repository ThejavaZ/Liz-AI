import pytest

from liz.core.notes import Note, NotesStore
from liz.infrastructure.persistence.notes_store import SQLiteNotesStore
from liz.tools.models import ToolStatus
from liz.tools.notes import Notes


class InMemoryNotesStore(NotesStore):
    def __init__(self) -> None:
        self.notes: list[Note] = []
        self._next_id = 1

    def add(self, title, content):
        note = Note(
            id=self._next_id,
            title=title,
            content=content,
            created_at="2026-01-01",
        )
        self.notes.append(note)
        self._next_id += 1
        return note

    def list(self, limit=20):
        return list(reversed(self.notes))[:limit]

    def search(self, query):
        q = query.lower()
        return [
            n
            for n in self.notes
            if q in n.title.lower() or q in n.content.lower()
        ]

    def delete(self, note_id):
        for index, note in enumerate(self.notes):
            if note.id == note_id:
                del self.notes[index]
                return True
        return False

    def close(self):
        pass


class TestSQLiteNotesStore:
    def _store(self, tmp_path) -> SQLiteNotesStore:
        return SQLiteNotesStore(tmp_path / "notes.db")

    def test_add_and_list(self, tmp_path) -> None:
        store = self._store(tmp_path)
        store.add("First", "content one")
        store.add("Second", "content two")

        notes = store.list()
        assert len(notes) == 2
        assert notes[0].title == "Second"
        assert notes[1].title == "First"
        store.close()

    def test_persists_across_reopen(self, tmp_path) -> None:
        self._store(tmp_path).add("Persisted", "body")
        self._store(tmp_path).close()

        reopened = self._store(tmp_path)
        assert reopened.list()[0].title == "Persisted"
        reopened.close()

    def test_search_title(self, tmp_path) -> None:
        store = self._store(tmp_path)
        store.add("Groceries", "milk, eggs")
        store.add("Work", "meeting notes")

        assert len(store.search("grocer")) == 1
        store.close()

    def test_search_content(self, tmp_path) -> None:
        store = self._store(tmp_path)
        store.add("Groceries", "milk, eggs")
        store.add("Work", "meeting notes")

        assert len(store.search("eggs")) == 1
        store.close()

    def test_search_no_match(self, tmp_path) -> None:
        store = self._store(tmp_path)
        store.add("Title", "content")
        assert store.search("zzz") == []
        store.close()

    def test_delete(self, tmp_path) -> None:
        store = self._store(tmp_path)
        note = store.add("Delete me", "x")

        assert store.delete(note.id) is True
        assert store.list() == []
        store.close()

    def test_delete_missing(self, tmp_path) -> None:
        store = self._store(tmp_path)
        assert store.delete(999) is False
        store.close()

    def test_creates_parent_dir(self, tmp_path) -> None:
        target = tmp_path / "a" / "b" / "notes.db"
        store = SQLiteNotesStore(target)
        assert target.exists()
        store.close()

    def test_list_limit(self, tmp_path) -> None:
        store = self._store(tmp_path)
        for i in range(10):
            store.add(f"Note {i}", "x")

        assert len(store.list(limit=3)) == 3
        store.close()


class TestNotesTool:
    def _tool(self) -> tuple[Notes, InMemoryNotesStore]:
        store = InMemoryNotesStore()
        return Notes(store), store

    def test_name(self) -> None:
        assert Notes(InMemoryNotesStore()).name == "notes"

    def test_description(self) -> None:
        assert "notes" in Notes(InMemoryNotesStore()).description.lower()

    def test_parameters_schema(self) -> None:
        schema = Notes(InMemoryNotesStore()).parameters_schema
        assert "action" in schema["properties"]
        assert "action" in schema["required"]

    def test_add_note(self) -> None:
        tool, store = self._tool()
        result = tool.execute(action="add", title="T", content="C")

        assert result.status == ToolStatus.SUCCESS
        assert "Saved note #1" in result.data
        assert len(store.notes) == 1

    def test_add_requires_title(self) -> None:
        tool, _ = self._tool()
        result = tool.execute(action="add", title="", content="C")
        assert result.status == ToolStatus.ERROR

    def test_add_requires_content(self) -> None:
        tool, _ = self._tool()
        result = tool.execute(action="add", title="T", content="")
        assert result.status == ToolStatus.ERROR

    def test_list_notes(self) -> None:
        tool, store = self._tool()
        store.add("One", "a")
        store.add("Two", "b")

        result = tool.execute(action="list")
        assert "One" in result.data
        assert "Two" in result.data

    def test_list_empty(self) -> None:
        tool, _ = self._tool()
        result = tool.execute(action="list")
        assert "No notes" in result.data

    def test_search_notes(self) -> None:
        tool, store = self._tool()
        store.add("Alpha", "x")
        store.add("Beta", "y")

        result = tool.execute(action="search", query="alpha")
        assert "Alpha" in result.data
        assert "Beta" not in result.data

    def test_search_requires_query(self) -> None:
        tool, _ = self._tool()
        result = tool.execute(action="search", query="")
        assert result.status == ToolStatus.ERROR

    def test_search_no_match(self) -> None:
        tool, store = self._tool()
        store.add("Alpha", "x")
        result = tool.execute(action="search", query="zzz")
        assert "No notes matching" in result.data

    def test_delete_note(self) -> None:
        tool, store = self._tool()
        store.add("Delete", "x")

        result = tool.execute(action="delete", note_id=1)
        assert "Deleted note #1" in result.data
        assert store.notes == []

    def test_delete_missing(self) -> None:
        tool, _ = self._tool()
        result = tool.execute(action="delete", note_id=999)
        assert result.status == ToolStatus.ERROR

    def test_delete_requires_id(self) -> None:
        tool, _ = self._tool()
        result = tool.execute(action="delete")
        assert result.status == ToolStatus.ERROR

    def test_unknown_action(self) -> None:
        tool, _ = self._tool()
        result = tool.execute(action="bogus")
        assert result.status == ToolStatus.ERROR
        assert "Unknown action" in result.error

    def test_store_error_reported(self) -> None:
        tool, store = self._tool()

        def boom(limit=20):
            raise RuntimeError("db down")

        store.list = boom

        result = tool.execute(action="list")
        assert result.status == ToolStatus.ERROR
