"""Notes tool.

Lets the model create, list, search and delete persistent notes. The
store is injected so the tool never opens a database itself.
"""

from __future__ import annotations

from typing import Any

from liz.core.notes import NotesStore
from liz.tools.base import Tool
from liz.tools.models import ToolResult


class Notes(Tool):
    def __init__(self, store: NotesStore) -> None:
        self.store = store

    @property
    def name(self) -> str:
        return "notes"

    @property
    def description(self) -> str:
        return "Create, list, search or delete persistent notes"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "search", "delete"],
                    "description": "What to do with notes",
                },
                "title": {
                    "type": "string",
                    "description": "Note title (for add)",
                },
                "content": {
                    "type": "string",
                    "description": "Note body (for add)",
                },
                "query": {
                    "type": "string",
                    "description": "Search text (for search)",
                },
                "note_id": {
                    "type": "integer",
                    "description": "Note id (for delete)",
                },
            },
            "required": ["action"],
        }

    def execute(
        self,
        action: str,
        title: str = "",
        content: str = "",
        query: str = "",
        note_id: int | None = None,
    ) -> ToolResult:
        handlers = {
            "add": self._add,
            "list": self._list,
            "search": self._search,
            "delete": self._delete,
        }

        handler = handlers.get(action)
        if handler is None:
            return ToolResult.error(
                f"Unknown action: {action}. Use add, list, search or delete."
            )

        try:
            return handler(title, content, query, note_id)
        except Exception as e:
            return ToolResult.error(f"Notes operation failed: {e}")

    def _add(self, title, content, query, note_id) -> ToolResult:
        if not title.strip():
            return ToolResult.error("Note title is required")
        if not content.strip():
            return ToolResult.error("Note content is required")

        note = self.store.add(title.strip(), content.strip())
        return ToolResult.success(f"Saved note #{note.id}: {note.title}")

    def _list(self, title, content, query, note_id) -> ToolResult:
        notes = self.store.list()
        if not notes:
            return ToolResult.success("No notes stored.")

        lines = [f"{n.id}. {n.title} ({n.created_at})" for n in notes]
        return ToolResult.success("\n".join(lines))

    def _search(self, title, content, query, note_id) -> ToolResult:
        if not query.strip():
            return ToolResult.error("Search query is required")

        notes = self.store.search(query.strip())
        if not notes:
            return ToolResult.success(f"No notes matching: {query}")

        lines = [f"{n.id}. {n.title}" for n in notes]
        return ToolResult.success("\n".join(lines))

    def _delete(self, title, content, query, note_id) -> ToolResult:
        if note_id is None:
            return ToolResult.error("note_id is required for delete")

        if self.store.delete(note_id):
            return ToolResult.success(f"Deleted note #{note_id}")
        return ToolResult.error(f"No note with id: {note_id}")
