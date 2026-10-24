import platform
import sys
from dataclasses import dataclass
from pathlib import Path

from liz.config.settings import Settings
from liz.core.agent import Agent
from liz.core.context import Context
from liz.core.notes import NotesStore
from liz.infrastructure.ai.base import AIProvider
from liz.infrastructure.ai.models_registry import (
    DEFAULT_MODELS,
    list_models,
    resolve_models,
)
from liz.tools.registry import ToolRegistry

API_KEY_FIELDS = {
    "gemini": "gemini_api_key",
    "openai": "openai_api_key",
    "claude": "anthropic_api_key",
    "deepseek": "deepseek_api_key",
}


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
        notes_store: NotesStore | None = None,
    ) -> None:
        self.settings = settings
        self.context = context
        self.tool_registry = tool_registry
        self.provider = provider
        self.agent = agent
        self.notes_store = notes_store
        self._commands: dict[str, callable] = {
            "help": self._help,
            "tools": self._tools,
            "models": self._models,
            "memory": self._memory,
            "notes": self._notes,
            "remember": self._remember,
            "forget": self._forget,
            "config": self._config,
            "doctor": self._doctor,
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
        args = parts[1] if len(parts) > 1 else ""

        handler = self._commands.get(command)
        if handler is None:
            return CommandResult(
                output=f"Unknown command: /{command}\nType /help for available commands."
            )

        return handler(args)

    def _help(self, args: str = "") -> CommandResult:
        text = """Available commands:
  /help     Show this help message
  /tools    List available tools
  /models   List available models for this provider
  /memory   Show stored facts and session info
  /remember Store a fact: /remember editor = neovim
  /forget   Remove a fact: /forget editor
  /notes    Manage notes: list | add <title> | del <id> | find <text>
  /config   Show effective configuration
  /doctor   Run health diagnostics
  /clear    Clear conversation history
  /status   Show current configuration
  /exit     Exit Liz AI
  /quit     Exit Liz AI"""
        return CommandResult(output=text)

    def _tools(self, args: str = "") -> CommandResult:
        tools = self.tool_registry.list_tools()
        if not tools:
            return CommandResult(output="No tools registered.")

        lines = ["Available tools:"]
        for tool in tools:
            lines.append(f"  {tool.name:20s} {tool.description}")
        return CommandResult(output="\n".join(lines))

    def _models(self, args: str = "") -> CommandResult:
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

    def _clear(self, args: str = "") -> CommandResult:
        if self.agent is not None:
            self.agent.clear_conversation()
        else:
            self.context.clear()
        return CommandResult(output="Conversation cleared.")

    def _memory(self, args: str = "") -> CommandResult:
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

    def _notes(self, args: str = "") -> CommandResult:
        if self.notes_store is None:
            return CommandResult(output="Notes are not enabled.")

        parts = args.split(maxsplit=1)
        sub = parts[0].lower() if parts else "list"
        rest = parts[1] if len(parts) > 1 else ""

        if sub == "list":
            notes = self.notes_store.list()
            if not notes:
                return CommandResult(output="No notes stored.")
            lines = [f"{n.id}. {n.title} ({n.created_at})" for n in notes]
            return CommandResult(output="\n".join(lines))

        if sub == "add":
            if not rest.strip():
                return CommandResult(
                    output="Usage: /notes add <title> (then describe it to Liz)"
                )
            note = self.notes_store.add(rest.strip(), "(created via /notes)")
            return CommandResult(output=f"Saved note #{note.id}: {note.title}")

        if sub in ("del", "delete", "rm"):
            try:
                note_id = int(rest.strip())
            except ValueError:
                return CommandResult(output="Usage: /notes del <id>")
            if self.notes_store.delete(note_id):
                return CommandResult(output=f"Deleted note #{note_id}")
            return CommandResult(output=f"No note with id: {note_id}")

        if sub in ("find", "search"):
            if not rest.strip():
                return CommandResult(output="Usage: /notes find <text>")
            notes = self.notes_store.search(rest.strip())
            if not notes:
                return CommandResult(output=f"No notes matching: {rest}")
            lines = [f"{n.id}. {n.title}" for n in notes]
            return CommandResult(output="\n".join(lines))

        return CommandResult(
            output="Usage: /notes [list|add <title>|del <id>|find <text>]"
        )
        
    def _remember(self, args: str = "") -> CommandResult:
        memory = self._memory_or_none()
        if memory is None:
            return CommandResult(output="Memory is not enabled.")

        fact = args.strip()
        if not fact:
            return CommandResult(output="Usage: /remember <clave> = <valor>")

        if "=" in fact:
            key, _, value = fact.partition("=")
        else:
            words = fact.split(maxsplit=1)
            key = words[0]
            value = words[1] if len(words) > 1 else "true"

        key, value = key.strip(), value.strip()
        if not key or not value:
            return CommandResult(output="Usage: /remember <clave> = <valor>")

        memory.learn_fact(key, value)
        return CommandResult(output=f"Remembered: {key} = {value}")

    def _forget(self, args: str = "") -> CommandResult:
        memory = self._memory_or_none()
        if memory is None:
            return CommandResult(output="Memory is not enabled.")

        key = args.strip()
        if not key:
            return CommandResult(output="Usage: /forget <clave>")

        if memory.forget_fact(key):
            return CommandResult(output=f"Forgot: {key}")
        return CommandResult(output=f"No fact stored as: {key}")

    def _memory_or_none(self):
        return getattr(self.agent, "memory", None) if self.agent else None

    def _config(self, args: str = "") -> CommandResult:
        rows = [
            ("ai_provider", self.settings.ai_provider),
            ("ai_model", self.settings.ai_model or "(auto)"),
            ("ollama_base_url", self.settings.ollama_base_url),
            ("data_dir", self.settings.data_dir),
        ]

        for label, field in API_KEY_FIELDS.items():
            value = getattr(self.settings, field, None)
            rows.append((field, self._mask(value)))

        lines = ["Configuration:"]
        for key, value in rows:
            lines.append(f"  {key:18s} {value}")
        return CommandResult(output="\n".join(lines))

    @staticmethod
    def _mask(value: str | None) -> str:
        if not value:
            return "(not set)"
        if len(value) <= 8:
            return "****"
        return f"{value[:4]}...{value[-4:]}"

    def _doctor(self, args: str = "") -> CommandResult:
        checks: list[tuple[str, str, str]] = []

        checks.append(("Python", "ok", sys.version.split()[0]))
        checks.append(("Platform", "ok", platform.system()))

        provider = self.settings.ai_provider.lower()
        key_field = API_KEY_FIELDS.get(provider)
        if key_field is None:
            checks.append(("Provider", "ok", f"{provider} (local, no key)"))
        else:
            has_key = bool(getattr(self.settings, key_field, None))
            status = "ok" if has_key else "warn"
            detail = "key set" if has_key else f"missing {key_field}"
            checks.append(("Provider", status, f"{provider}: {detail}"))

        try:
            chain = resolve_models(provider, self.settings.ai_model)
            checks.append(
                ("Models", "ok", f"{len(chain)} in fallback chain")
            )
        except ValueError as error:
            checks.append(("Models", "fail", str(error)))

        if provider == "ollama":
            checks.append(self._check_ollama())

        tools = self.tool_registry.list_tools()
        checks.append(("Tools", "ok", f"{len(tools)} registered"))

        memory = getattr(self.agent, "memory", None) if self.agent else None
        if memory is not None:
            db_path = Path(self.settings.data_dir) / "liz.db"
            try:
                memory.store.get_facts()
                checks.append(("Memory", "ok", f"db at {db_path}"))
            except Exception as error:
                checks.append(("Memory", "fail", str(error)))
        else:
            checks.append(("Memory", "warn", "not enabled"))

        if self.notes_store is not None:
            try:
                self.notes_store.list(limit=1)
                checks.append(("Notes", "ok", "store reachable"))
            except Exception as error:
                checks.append(("Notes", "fail", str(error)))

        checks.append(self._check_web_search())

        icons = {"ok": "[ok]  ", "warn": "[warn]", "fail": "[FAIL]"}
        lines = ["Liz health check", ""]
        for name, status, detail in checks:
            lines.append(f"{icons[status]} {name:10s} {detail}")

        failed = sum(1 for _, status, _ in checks if status == "fail")
        warned = sum(1 for _, status, _ in checks if status == "warn")
        lines.append("")
        if failed:
            lines.append(f"{failed} problem(s) found.")
        elif warned:
            lines.append(f"Healthy with {warned} warning(s).")
        else:
            lines.append("All checks passed.")

        return CommandResult(output="\n".join(lines))

    def _check_ollama(self) -> tuple[str, str, str]:
        import urllib.error
        import urllib.request

        base = self.settings.ollama_base_url.rstrip("/").rsplit("/v1", 1)[0]
        try:
            with urllib.request.urlopen(f"{base}/api/tags", timeout=3):
                return ("Ollama", "ok", f"reachable at {base}")
        except urllib.error.URLError:
            return ("Ollama", "fail", f"not reachable at {base}")
        except Exception as error:
            return ("Ollama", "fail", str(error))

    @staticmethod
    def _check_web_search() -> tuple[str, str, str]:
        try:
            import duckduckgo_search  # noqa: F401

            return ("WebSearch", "ok", "duckduckgo_search installed")
        except ImportError:
            return ("WebSearch", "warn", "using HTML fallback")

    def _active_model(self) -> str:
        provider = self.provider
        if provider is None:
            return self.settings.ai_model or "unknown"

        active = getattr(provider, "active", provider)
        return getattr(active, "model", self.settings.ai_model or "unknown")

    def _status(self, args: str = "") -> CommandResult:
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

    def _exit(self, args: str = "") -> CommandResult:
        return CommandResult(output="Goodbye!", should_exit=True)
