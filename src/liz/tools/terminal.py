import subprocess
import shlex
from pathlib import Path
from typing import Any, Sequence

from liz.tools.base import Tool
from liz.tools.models import ToolResult
from liz.tools.sandbox import format_roots, within_roots


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

# find flags that run other programs or delete files
BLOCKED_ARGUMENTS = frozenset({
    "-exec",
    "-execdir",
    "-ok",
    "-okdir",
    "-delete",
})


class RunCommand(Tool):
    def __init__(
        self,
        allowed_commands: frozenset[str] | None = None,
        roots: Sequence[Path | str] | None = None,
    ) -> None:
        self._allowed = allowed_commands or DEFAULT_ALLOWED_COMMANDS
        self.roots = list(roots) if roots else None

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

        denied = self._check_arguments(parts[1:])
        if denied is not None:
            return denied

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

    def _check_arguments(self, args: list[str]) -> ToolResult | None:
        """Reject dangerous flags and paths outside the sandbox."""
        for arg in args:
            if arg in BLOCKED_ARGUMENTS:
                return ToolResult.error(
                    f"Argument not allowed: {arg} (it can run other "
                    "programs or delete files)"
                )

        for arg in args:
            if arg.startswith("-") or not arg:
                continue

            # arguments that name a location: absolute, home or relative
            if arg[0] in "/~.":
                allowed, resolved = within_roots(Path(arg), self.roots)
                if not allowed:
                    return ToolResult.error(
                        f"Path outside allowed directories: {resolved}. "
                        f"Allowed: {format_roots(self.roots)} "
                        "(extend with FS_ROOTS)"
                    )
        return None
