import pytest
import tempfile
from pathlib import Path

from liz.tools.filesystem import ListDirectory, ReadFile, FileExists
from liz.tools.models import ToolStatus


class TestListDirectory:
    def test_list_existing_directory(self, tmp_path: Path) -> None:
        (tmp_path / "file1.txt").touch()
        (tmp_path / "file2.txt").touch()
        (tmp_path / "subdir").mkdir()

        tool = ListDirectory()
        result = tool.execute(path=str(tmp_path))

        assert result.status == ToolStatus.SUCCESS
        assert "file1.txt" in result.data
        assert "file2.txt" in result.data
        assert "subdir/" in result.data

    def test_list_empty_directory(self, tmp_path: Path) -> None:
        tool = ListDirectory()
        result = tool.execute(path=str(tmp_path))

        assert result.status == ToolStatus.SUCCESS
        assert "empty" in result.data.lower()

    def test_list_nonexistent_path(self) -> None:
        tool = ListDirectory()
        result = tool.execute(path="/nonexistent/path")

        assert result.status == ToolStatus.ERROR
        assert "does not exist" in result.error

    def test_list_file_not_directory(self, tmp_path: Path) -> None:
        file_path = tmp_path / "file.txt"
        file_path.touch()

        tool = ListDirectory()
        result = tool.execute(path=str(file_path))

        assert result.status == ToolStatus.ERROR
        assert "not a directory" in result.error

    def test_list_default_path(self) -> None:
        tool = ListDirectory()
        result = tool.execute()
        assert result.status == ToolStatus.SUCCESS


class TestReadFile:
    def test_read_existing_file(self, tmp_path: Path) -> None:
        file_path = tmp_path / "test.txt"
        file_path.write_text("Hello, World!")

        tool = ReadFile()
        result = tool.execute(path=str(file_path))

        assert result.status == ToolStatus.SUCCESS
        assert result.data == "Hello, World!"

    def test_read_nonexistent_file(self) -> None:
        tool = ReadFile()
        result = tool.execute(path="/nonexistent/file.txt")

        assert result.status == ToolStatus.ERROR
        assert "does not exist" in result.error

    def test_read_directory_not_file(self, tmp_path: Path) -> None:
        tool = ReadFile()
        result = tool.execute(path=str(tmp_path))

        assert result.status == ToolStatus.ERROR
        assert "not a file" in result.error

    def test_read_multiline_file(self, tmp_path: Path) -> None:
        file_path = tmp_path / "multi.txt"
        file_path.write_text("line1\nline2\nline3")

        tool = ReadFile()
        result = tool.execute(path=str(file_path))

        assert result.status == ToolStatus.SUCCESS
        assert "line1" in result.data
        assert "line2" in result.data
        assert "line3" in result.data


class TestFileExists:
    def test_existing_file(self, tmp_path: Path) -> None:
        file_path = tmp_path / "exists.txt"
        file_path.touch()

        tool = FileExists()
        result = tool.execute(path=str(file_path))

        assert result.status == ToolStatus.SUCCESS
        assert "file" in result.data.lower()

    def test_existing_directory(self, tmp_path: Path) -> None:
        tool = FileExists()
        result = tool.execute(path=str(tmp_path))

        assert result.status == ToolStatus.SUCCESS
        assert "directory" in result.data.lower()

    def test_nonexistent_path(self) -> None:
        tool = FileExists()
        result = tool.execute(path="/nonexistent/path")

        assert result.status == ToolStatus.SUCCESS
        assert "not exist" in result.data.lower()
