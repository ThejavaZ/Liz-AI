"""Memory tool.

Lets the model store and forget facts on its own, so "remember that my
editor is Neovim" survives across sessions without a manual command.
"""

from __future__ import annotations

from typing import Any

from liz.core.memory import ConversationMemory
from liz.tools.base import Tool
from liz.tools.models import ToolResult


class Memory(Tool):
    def __init__(self, memory: ConversationMemory) -> None:
        self.memory = memory

    @property
    def name(self) -> str:
        return "memory"

    @property
    def description(self) -> str:
        return "Remember, forget or search facts and past conversation that persist across sessions"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["remember", "forget", "list", "recall"],
                    "description": "What to do",
                },
                "key": {
                    "type": "string",
                    "description": "Short fact name, e.g. 'editor' or 'user_name'",
                },
                "value": {
                    "type": "string",
                    "description": "Fact value (for remember)",
                },
                "query": {
                    "type": "string",
                    "description": "Text to search for (for recall)",
                },
            },
            "required": ["action"],
        }

    def execute(
        self,
        action: str,
        key: str = "",
        value: str = "",
        query: str = "",
    ) -> ToolResult:
        handlers = {
            "remember": self._remember,
            "forget": self._forget,
            "list": self._list,
            "recall": self._recall,
        }

        handler = handlers.get(action)
        if handler is None:
            return ToolResult.error(
                "Unknown action: use remember, forget, list or recall."
            )

        try:
            return handler(key.strip(), value.strip(), query.strip())
        except Exception as e:
            return ToolResult.error(f"Memory operation failed: {e}")

    def _remember(self, key: str, value: str, query: str = "") -> ToolResult:
        if not key:
            return ToolResult.error("A fact key is required")
        if not value:
            return ToolResult.error("A fact value is required")

        self.memory.learn_fact(key, value)
        return ToolResult.success(f"Remembered: {key} = {value}")

    def _forget(self, key: str, value: str, query: str = "") -> ToolResult:
        if not key:
            return ToolResult.error("A fact key is required")

        if self.memory.forget_fact(key):
            return ToolResult.success(f"Forgot: {key}")
        return ToolResult.error(f"No fact stored as: {key}")

    def _list(self, key: str, value: str, query: str = "") -> ToolResult:
        facts = self.memory.facts()
        if not facts:
            return ToolResult.success("No facts stored yet.")

        lines = [f"{k} = {v}" for k, v in facts.items()]
        return ToolResult.success("\n".join(lines))

    def _recall(self, key: str, value: str, query: str = "") -> ToolResult:
        if not query:
            return ToolResult.error("A search query is required")

        facts, messages = self.memory.recall(query, all_sessions=True)

        if not facts and not messages:
            return ToolResult.success(f"Nothing found for: {query}")

        lines: list[str] = []
        if facts:
            lines.append("Facts:")
            lines.extend(f"  {k} = {v}" for k, v in facts.items())

        if messages:
            if lines:
                lines.append("")
            lines.append("Messages:")
            lines.extend(f"  [{m.role}] {m.content}" for m in messages)

        return ToolResult.success("\n".join(lines))
