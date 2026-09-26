import pytest
from unittest.mock import MagicMock

from liz.tools.base import Tool
from liz.tools.registry import ToolRegistry
from liz.tools.permissions import PermissionLayer, Permission
from liz.tools.models import ToolResult, ToolStatus


class MockTool(Tool):
    def __init__(self, tool_name: str = "mock_tool") -> None:
        self._name = tool_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "A mock tool for testing"

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult.success(f"executed {self._name}")


class FailingTool(Tool):
    @property
    def name(self) -> str:
        return "failing_tool"

    @property
    def description(self) -> str:
        return "A tool that always fails"

    def execute(self, **kwargs) -> ToolResult:
        return ToolResult.error("always fails")


class TestToolRegistry:
    def test_register_and_get(self) -> None:
        registry = ToolRegistry()
        tool = MockTool("test")
        registry.register(tool)

        assert registry.get("test") is tool
        assert registry.get("nonexistent") is None

    def test_list_tools(self) -> None:
        registry = ToolRegistry()
        t1 = MockTool("t1")
        t2 = MockTool("t2")
        registry.register(t1)
        registry.register(t2)

        tools = registry.list_tools()
        assert len(tools) == 2
        assert t1 in tools
        assert t2 in tools

    def test_list_names(self) -> None:
        registry = ToolRegistry()
        registry.register(MockTool("alpha"))
        registry.register(MockTool("beta"))

        names = registry.list_names()
        assert "alpha" in names
        assert "beta" in names

    def test_execute_unknown_tool(self) -> None:
        registry = ToolRegistry()
        result = registry.execute("nonexistent")

        assert result.status == ToolStatus.ERROR
        assert "Unknown tool" in result.error

    def test_execute_allowed_tool(self) -> None:
        permissions = PermissionLayer()
        permissions.set_rule("mock_tool", Permission.ALLOW)

        registry = ToolRegistry(permissions=permissions)
        registry.register(MockTool("mock_tool"))

        result = registry.execute("mock_tool")
        assert result.status == ToolStatus.SUCCESS
        assert "executed" in result.data

    def test_execute_denied_tool(self) -> None:
        permissions = PermissionLayer()
        permissions.set_rule("mock_tool", Permission.DENY)

        registry = ToolRegistry(permissions=permissions)
        registry.register(MockTool("mock_tool"))

        result = registry.execute("mock_tool")
        assert result.status == ToolStatus.DENIED
        assert "denied" in result.error.lower()

    def test_execute_ask_tool_rejected(self) -> None:
        permissions = PermissionLayer()
        permissions.set_rule("mock_tool", Permission.ASK)

        registry = ToolRegistry(permissions=permissions)
        registry.register(MockTool("mock_tool"))

        result = registry.execute("mock_tool")
        assert result.status == ToolStatus.DENIED

    def test_default_permission_is_ask(self) -> None:
        registry = ToolRegistry()
        registry.register(MockTool("mock_tool"))

        result = registry.execute("mock_tool")
        assert result.status == ToolStatus.DENIED
