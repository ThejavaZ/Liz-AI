import os
from pathlib import Path

from liz.tools.base import Tool
from liz.tools.models import ToolResult


class ListDirectory(Tool):
    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def description(self) -> str:
        return "List files and directories at a given path"

    def execute(self, path: str = ".") -> ToolResult:
        try:
            target = Path(path)
            if not target.exists():
                return ToolResult.error(f"Path does not exist: {path}")
            if not target.is_dir():
                return ToolResult.error(f"Path is not a directory: {path}")

            entries = sorted(os.listdir(target))
            if not entries:
                return ToolResult.success(f"Directory is empty: {path}")

            lines = []
            for entry in entries:
                full_path = target / entry
                if full_path.is_dir():
                    lines.append(f"  {entry}/")
                else:
                    lines.append(f"  {entry}")

            return ToolResult.success("\n".join(lines))
        except Exception as e:
            return ToolResult.error(f"Error listing directory: {e}")


class ReadFile(Tool):
    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file"

    def execute(self, path: str) -> ToolResult:
        try:
            target = Path(path)
            if not target.exists():
                return ToolResult.error(f"File does not exist: {path}")
            if not target.is_file():
                return ToolResult.error(f"Path is not a file: {path}")

            content = target.read_text(encoding="utf-8")
            return ToolResult.success(content)
        except UnicodeDecodeError:
            return ToolResult.error(f"Cannot read binary file: {path}")
        except Exception as e:
            return ToolResult.error(f"Error reading file: {e}")


class FileExists(Tool):
    @property
    def name(self) -> str:
        return "file_exists"

    @property
    def description(self) -> str:
        return "Check if a file or directory exists"

    def execute(self, path: str) -> ToolResult:
        try:
            target = Path(path)
            if target.exists():
                file_type = "directory" if target.is_dir() else "file"
                return ToolResult.success(f"Exists ({file_type}): {path}")
            return ToolResult.success(f"Does not exist: {path}")
        except Exception as e:
            return ToolResult.error(f"Error checking path: {e}")
