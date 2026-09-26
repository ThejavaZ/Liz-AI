import pytest
from unittest.mock import patch, MagicMock
import subprocess

from liz.tools.git import GitStatus, GitLog, GitBranch
from liz.tools.models import ToolStatus


class TestGitStatus:
    @patch("liz.tools.git.subprocess.run")
    def test_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0, stdout="M file.py\n", stderr=""
        )
        tool = GitStatus()
        result = tool.execute(cwd="/test")

        assert result.status == ToolStatus.SUCCESS
        assert "M file.py" in result.data

    @patch("liz.tools.git.subprocess.run")
    def test_clean_working_tree(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        tool = GitStatus()
        result = tool.execute()

        assert result.status == ToolStatus.SUCCESS
        assert "clean" in result.data.lower()

    @patch("liz.tools.git.subprocess.run")
    def test_git_error(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=128, stdout="", stderr="fatal: not a git repo"
        )
        tool = GitStatus()
        result = tool.execute()

        assert result.status == ToolStatus.ERROR
        assert "fatal" in result.error

    @patch("liz.tools.git.subprocess.run")
    def test_git_not_installed(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = FileNotFoundError
        tool = GitStatus()
        result = tool.execute()

        assert result.status == ToolStatus.ERROR
        assert "not installed" in result.error


class TestGitLog:
    @patch("liz.tools.git.subprocess.run")
    def test_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0, stdout="abc1234 Initial commit\n", stderr=""
        )
        tool = GitLog()
        result = tool.execute()

        assert result.status == ToolStatus.SUCCESS
        assert "abc1234" in result.data

    @patch("liz.tools.git.subprocess.run")
    def test_empty_log(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        tool = GitLog()
        result = tool.execute()

        assert result.status == ToolStatus.SUCCESS
        assert "no commits" in result.data.lower()

    @patch("liz.tools.git.subprocess.run")
    def test_git_error(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=128, stdout="", stderr="fatal: error"
        )
        tool = GitLog()
        result = tool.execute()

        assert result.status == ToolStatus.ERROR


class TestGitBranch:
    @patch("liz.tools.git.subprocess.run")
    def test_success(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=0, stdout="* main\n  dev\n", stderr=""
        )
        tool = GitBranch()
        result = tool.execute()

        assert result.status == ToolStatus.SUCCESS
        assert "main" in result.data

    @patch("liz.tools.git.subprocess.run")
    def test_no_branches(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        tool = GitBranch()
        result = tool.execute()

        assert result.status == ToolStatus.SUCCESS
        assert "no branches" in result.data.lower()

    @patch("liz.tools.git.subprocess.run")
    def test_git_error(self, mock_run: MagicMock) -> None:
        mock_run.return_value = MagicMock(
            returncode=128, stdout="", stderr="fatal: not a git repo"
        )
        tool = GitBranch()
        result = tool.execute()

        assert result.status == ToolStatus.ERROR
