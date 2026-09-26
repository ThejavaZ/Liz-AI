import pytest
from unittest.mock import patch, MagicMock

from liz.tools.terminal import RunCommand
from liz.tools.models import ToolStatus


class TestRunCommand:
    @patch("liz.tools.terminal.subprocess.run")
    def test_allowed_command(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0, stdout="hello\n", stderr=""
        )
        tool = RunCommand()
        result = tool.execute(command="echo hello")

        assert result.status == ToolStatus.SUCCESS
        assert "hello" in result.data

    @patch("liz.tools.terminal.subprocess.run")
    def test_disallowed_command(self, mock_run: MagicMock) -> None:
        tool = RunCommand()
        result = tool.execute(command="rm -rf /")

        assert result.status == ToolStatus.ERROR
        assert "not allowed" in result.error.lower()
        mock_run.assert_not_called()

    @patch("liz.tools.terminal.subprocess.run")
    def test_command_failure(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=1, stdout="", stderr="error occurred"
        )
        tool = RunCommand()
        result = tool.execute(command="ls /nonexistent")

        assert result.status == ToolStatus.ERROR
        assert "failed" in result.error.lower()

    @patch("liz.tools.terminal.subprocess.run")
    def test_command_not_installed(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError
        tool = RunCommand(allowed_commands=frozenset({"nonexistent"}))
        result = tool.execute(command="nonexistent")

        assert result.status == ToolStatus.ERROR

    @patch("liz.tools.terminal.subprocess.run")
    def test_custom_allowed_list(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
        custom_allowed = frozenset({"mycommand"})
        tool = RunCommand(allowed_commands=custom_allowed)

        result = tool.execute(command="mycommand")
        assert result.status == ToolStatus.SUCCESS

        result = tool.execute(command="othercommand")
        assert result.status == ToolStatus.ERROR
        assert "not allowed" in result.error.lower()

    def test_empty_command(self) -> None:
        tool = RunCommand()
        result = tool.execute(command="")
        assert result.status == ToolStatus.ERROR
