import pytest
from liz.tools.models import ToolResult, ToolStatus


class TestToolResult:
    def test_success(self) -> None:
        result = ToolResult.success("data")
        assert result.status == ToolStatus.SUCCESS
        assert result.data == "data"
        assert result.error is None

    def test_error(self) -> None:
        result = ToolResult.error("something failed")
        assert result.status == ToolStatus.ERROR
        assert result.data == ""
        assert result.error == "something failed"

    def test_denied(self) -> None:
        result = ToolResult.denied("no permission")
        assert result.status == ToolStatus.DENIED
        assert result.data == ""
        assert result.error == "no permission"
