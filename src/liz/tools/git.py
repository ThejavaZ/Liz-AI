import subprocess

from liz.tools.base import Tool
from liz.tools.models import ToolResult


class GitStatus(Tool):
    @property
    def name(self) -> str:
        return "git_status"

    @property
    def description(self) -> str:
        return "Show the working tree status"

    def execute(self, cwd: str | None = None) -> ToolResult:
        try:
            result = subprocess.run(
                ["git", "status", "--short"],
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=10,
            )
            if result.returncode != 0:
                return ToolResult.error(result.stderr.strip())
            output = result.stdout.strip()
            return ToolResult.success(output if output else "Working tree clean")
        except FileNotFoundError:
            return ToolResult.error("Git is not installed")
        except subprocess.TimeoutExpired:
            return ToolResult.error("Git command timed out")
        except Exception as e:
            return ToolResult.error(f"Error running git status: {e}")


class GitLog(Tool):
    @property
    def name(self) -> str:
        return "git_log"

    @property
    def description(self) -> str:
        return "Show recent git commit log"

    def execute(self, cwd: str | None = None, max_count: int = 10) -> ToolResult:
        try:
            result = subprocess.run(
                ["git", "log", f"--max-count={max_count}", "--oneline"],
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=10,
            )
            if result.returncode != 0:
                return ToolResult.error(result.stderr.strip())
            output = result.stdout.strip()
            return ToolResult.success(output if output else "No commits found")
        except FileNotFoundError:
            return ToolResult.error("Git is not installed")
        except subprocess.TimeoutExpired:
            return ToolResult.error("Git command timed out")
        except Exception as e:
            return ToolResult.error(f"Error running git log: {e}")


class GitBranch(Tool):
    @property
    def name(self) -> str:
        return "git_branch"

    @property
    def description(self) -> str:
        return "List git branches"

    def execute(self, cwd: str | None = None) -> ToolResult:
        try:
            result = subprocess.run(
                ["git", "branch", "--list"],
                capture_output=True,
                text=True,
                cwd=cwd,
                timeout=10,
            )
            if result.returncode != 0:
                return ToolResult.error(result.stderr.strip())
            output = result.stdout.strip()
            return ToolResult.success(output if output else "No branches found")
        except FileNotFoundError:
            return ToolResult.error("Git is not installed")
        except subprocess.TimeoutExpired:
            return ToolResult.error("Git command timed out")
        except Exception as e:
            return ToolResult.error(f"Error running git branch: {e}")
