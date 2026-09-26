from liz.tools.base import Tool
from liz.tools.models import ToolResult, ToolStatus
from liz.tools.permissions import PermissionLayer, Permission


class ToolRegistry:
    def __init__(self, permissions: PermissionLayer | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self.permissions = permissions or PermissionLayer()

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        return list(self._tools.keys())

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult.error(f"Unknown tool: {tool_name}")

        permission = self.permissions.check(tool_name)

        if permission == Permission.DENY:
            return ToolResult.denied(f"Permission denied for tool: {tool_name}")

        if permission == Permission.ASK:
            if not self._confirm_execution(tool_name):
                return ToolResult.denied(f"Permission denied for tool: {tool_name}")

        return tool.execute(**kwargs)

    def _confirm_execution(self, tool_name: str) -> bool:
        return False
