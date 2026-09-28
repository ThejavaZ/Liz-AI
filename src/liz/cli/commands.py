from dataclasses import dataclass

from liz.config.settings import Settings
from liz.core.context import Context
from liz.tools.registry import ToolRegistry


@dataclass
class CommandResult:
    output: str
    should_exit: bool = False


class CommandHandler:
    def __init__(
        self,
        settings: Settings,
        context: Context,
        tool_registry: ToolRegistry,
    ) -> None:
        self.settings = settings
        self.context = context
        self.tool_registry = tool_registry
        self._commands: dict[str, callable] = {
            "help": self._help,
            "tools": self._tools,
            "clear": self._clear,
            "status": self._status,
            "exit": self._exit,
            "quit": self._exit,
        }

    def handle(self, user_input: str) -> CommandResult | None:
        if not user_input.startswith("/"):
            return None

        parts = user_input.strip().split(maxsplit=1)
        command = parts[0][1:].lower()

        handler = self._commands.get(command)
        if handler is None:
            return CommandResult(
                output=f"Unknown command: /{command}\nType /help for available commands."
            )

        return handler()

    def _help(self) -> CommandResult:
        text = """Available commands:
  /help    Show this help message
  /tools   List available tools
  /clear   Clear conversation history
  /status  Show current configuration
  /exit    Exit Liz AI
  /quit    Exit Liz AI"""
        return CommandResult(output=text)

    def _tools(self) -> CommandResult:
        tools = self.tool_registry.list_tools()
        if not tools:
            return CommandResult(output="No tools registered.")

        lines = ["Available tools:"]
        for tool in tools:
            lines.append(f"  {tool.name:20s} {tool.description}")
        return CommandResult(output="\n".join(lines))

    def _clear(self) -> CommandResult:
        self.context.clear()
        return CommandResult(output="Conversation cleared.")

    def _status(self) -> CommandResult:
        tools_count = len(self.tool_registry.list_tools())
        messages_count = len(self.context.get_messages()) - 1
        text = f"""Liz AI Status:
  Provider:  {self.settings.ai_provider}
  Model:     {self.settings.ai_model}
  Tools:     {tools_count} registered
  Messages:  {messages_count} in context"""
        return CommandResult(output=text)

    def _exit(self) -> CommandResult:
        return CommandResult(output="Goodbye!", should_exit=True)
