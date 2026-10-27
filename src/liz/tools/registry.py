from typing import Any, Callable

from liz.tools.base import Tool
from liz.tools.models import ToolResult, ToolStatus
from liz.tools.permissions import PermissionLayer, Permission

ConfirmCallback = Callable[[str, str], bool]
"""``(tool_name, preview) -> approved``, provided by the CLI."""


class ToolRegistry:
    def __init__(
        self,
        permissions: PermissionLayer | None = None,
        confirm: ConfirmCallback | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self.permissions = permissions or PermissionLayer()
        # Without a way to ask the user, ASK must deny (safe default).
        self.confirm = confirm

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_schema,
            })
        return schemas

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult.error(f"Unknown tool: {tool_name}")

        action = kwargs.get("action")
        action = action if isinstance(action, str) else None
        permission = self.permissions.check(tool_name, action)

        if permission == Permission.DENY:
            return ToolResult.denied(f"Permission denied for tool: {tool_name}")

        if permission == Permission.ASK:
            if not self._request_confirmation(tool_name, kwargs):
                return ToolResult.denied(
                    f"Permission denied for tool: {tool_name}"
                )

        try:
            return tool.execute(**kwargs)
        except Exception as exc:
            # A crashing tool is an error result, never an exception
            # through the agent loop: an unhandled raise would abort the
            # round halfway and orphan the remaining tool_calls.
            return ToolResult.error(f"{tool_name} failed: {exc!r}")

    def _request_confirmation(
        self, tool_name: str, kwargs: dict[str, Any]
    ) -> bool:
        if self.confirm is None:
            return False
        try:
            return bool(self.confirm(tool_name, self._preview(kwargs)))
        except Exception:
            # a broken prompt (EOF, Ctrl-C) must never approve a request
            return False

    @staticmethod
    def _preview(kwargs: dict[str, Any]) -> str:
        """Compact, human readable summary of the request arguments."""
        parts: list[str] = []
        for key, value in kwargs.items():
            if value is None or value == "":
                continue
            if not isinstance(value, (str, int, float, bool)):
                continue
            text = str(value)
            if len(text) > 80:
                text = text[:77] + "..."
            parts.append(f"{key}={text!r}")
        return ", ".join(parts)
