import subprocess
import shlex
from typing import Any

from liz.tools.base import Tool
from liz.tools.models import ToolResult


DEFAULT_ALLOWED_COMMANDS = frozenset({
    "ls",
    "pwd",
    "whoami",
    "date",
    "echo",
    "cat",
    "head",
    "tail",
    "wc",
    "grep",
    "find",
    "file",
    "du",
    "df",
    "uname",
    "hostname",
})


class RunCommand(Tool):
    def __init__(self, allowed_commands: frozenset[str] | None = None) -> None:
        self._allowed = allowed_commands or DEFAULT_ALLOWED_COMMANDS

    @property
    def name(self) -> str:
        return "run_command"

    @property
    def description(self) -> str:
        return "Execute a safe system command from the allowed list"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to execute",
                }
            },
            "required": ["command"],
        }

    @property
    def requires_confirmation(self) -> bool:
        return False

    def execute(self, command: str, cwd: str | None = None) -> ToolResult:
        try:
            parts = shlex.split(command)
        except ValueError as e:
            return ToolResult.error(f"Invalid command syntax: {e}")

        if not parts:
            return ToolResult.error("Empty command")

        cmd_name = parts[0]

        if cmd_name not in self._allowed:
            return ToolResult.error(
                f"Command not allowed: {cmd_name}. "
                f"Allowed: {', '.join(sorted(self._allowed))}"
            )

        try:
            result = subprocess.run(
                parts,
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=30,
            )
            output = result.stdout
            if result.returncode != 0:
                return ToolResult.error(
                    f"Command failed (exit {result.returncode}):\n{result.stderr or output}"
                )
            return ToolResult.success(output.strip() if output else "Command executed successfully")
        except subprocess.TimeoutExpired:
            return ToolResult.error("Command timed out after 30 seconds")
        except Exception as e:
            return ToolResult.error(f"Error executing command: {e}")
