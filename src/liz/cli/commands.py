from dataclasses import dataclass

from liz.config.settings import Settings
from liz.core.agent import Agent
from liz.core.context import Context
from liz.infrastructure.ai.base import AIProvider
from liz.infrastructure.ai.models_registry import (
    DEFAULT_MODELS,
    list_models,
)
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
        provider: AIProvider | None = None,
        agent: Agent | None = None,
    ) -> None:
        self.settings = settings
        self.context = context
        self.tool_registry = tool_registry
        self.provider = provider
        self.agent = agent
        self._commands: dict[str, callable] = {
            "help": self._help,
            "tools": self._tools,
            "models": self._models,
            "memory": self._memory,
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
  /models  List available models for this provider
  /memory  Show stored facts and session info
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

    def _models(self) -> CommandResult:
        provider = self.settings.ai_provider.lower()
        known = list_models(provider)
        if not known:
            return CommandResult(
                output=f"No model list available for provider: {provider}"
            )

        default = DEFAULT_MODELS.get(provider, "unknown")
        requested = self.settings.ai_model
        lines = [f"Models for {provider}:"]
        for name in known:
            markers = []
            if name == default:
                markers.append("default")
            if name == requested:
                markers.append("requested")
            suffix = f"  ({', '.join(markers)})" if markers else ""
            lines.append(f"  {name}{suffix}")

        if not requested:
            lines.append("\nNo model requested, using the default chain.")
        return CommandResult(output="\n".join(lines))

    def _clear(self) -> CommandResult:
        if self.agent is not None:
            self.agent.clear_conversation()
        else:
            self.context.clear()
        return CommandResult(output="Conversation cleared.")

    def _memory(self) -> CommandResult:
        memory = getattr(self.agent, "memory", None) if self.agent else None
        if memory is None:
            return CommandResult(output="Memory is not enabled.")

        facts = memory.facts()
        history = memory.load_history()
        lines = [
            f"Session:  {memory.session_id}",
            f"Messages: {len(history)} restored from disk",
            f"Facts:    {len(facts)} stored",
        ]
        if facts:
            lines.append("")
            lines.append("Known facts:")
            for key, value in facts.items():
                lines.append(f"  {key} = {value}")
        return CommandResult(output="\n".join(lines))

    def _active_model(self) -> str:
        provider = self.provider
        if provider is None:
            return self.settings.ai_model or "unknown"

        active = getattr(provider, "active", provider)
        return getattr(active, "model", self.settings.ai_model or "unknown")

    def _status(self) -> CommandResult:
        tools_count = len(self.tool_registry.list_tools())
        messages_count = len(self.context.get_messages()) - 1
        fallbacks = len(getattr(self.provider, "providers", [])) - 1
        text = f"""Liz AI Status:
  Provider:  {self.settings.ai_provider}
  Model:     {self._active_model()}
  Fallbacks: {max(fallbacks, 0)} available
  Tools:     {tools_count} registered
  Messages:  {messages_count} in context"""
        return CommandResult(output=text)

    def _exit(self) -> CommandResult:
        return CommandResult(output="Goodbye!", should_exit=True)
