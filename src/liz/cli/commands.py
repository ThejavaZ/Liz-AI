import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

from liz.config.settings import Settings
from liz.core.agent import Agent
from liz.core.context import Context
from liz.core.notes import NotesStore
from liz.core.voice import VoiceError
from liz.cli.voice import VoiceLoop
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
        voice: VoiceLoop | None = None,
    ) -> None:
        self.settings = settings
        self.context = context
        self.tool_registry = tool_registry
        self.provider = provider
        self.agent = agent
        self.notes_store = notes_store
        self.voice = voice
        self._commands: dict[str, callable] = {
            "help": self._help,
            "tools": self._tools,
            "models": self._models,
            "memory": self._memory,
            "notes": self._notes,
            "remember": self._remember,
            "forget": self._forget,
            "session": self._session,
            "recall": self._recall,
            "export": self._export,
            "import": self._import,
            "voice": self._voice,
            "talk": self._talk,
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
  /session  List sessions, or /session <name> to switch
  /recall   Search past messages and facts: /recall <texto>
  /export   Save history to JSON: /export [archivo.json]
  /import   Load history from JSON: /import <archivo.json>
  /notes    Manage notes: list | add <title> | del <id> | find <text>
  /talk     One voice turn: record, transcribe, answer aloud
  /voice    Voice status, or /voice on | off for the background listener
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
            permission = self.tool_registry.permissions.check(tool.name)
            lines.append(
                f"  {tool.name:20s} [{permission.value:5s}] {tool.description}"
            )
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

    def _session(self, args: str = "") -> CommandResult:
        memory = self._memory_or_none()
        if memory is None:
            return CommandResult(output="Memory is not enabled.")

        name = args.strip()
        if not name:
            sessions = memory.sessions()
            lines = ["Sessions:"]
            for session in sessions:
                marker = " (current)" if session == memory.session_id else ""
                lines.append(f"  {session}{marker}")
            lines.append("\nUsage: /session <name> to switch or create")
            return CommandResult(output="\n".join(lines))

        if not self._valid_session_name(name):
            return CommandResult(
                output="Session names may only use letters, numbers, - and _"
            )

        self.agent.switch_session(name)
        count = len(self.agent.context.get_messages()) - 1
        return CommandResult(
            output=f"Switched to session '{name}' ({count} messages in context)"
        )

    @staticmethod
    def _valid_session_name(name: str) -> bool:
        return len(name) <= 40 and all(
            ch.isalnum() or ch in "-_" for ch in name
        )

    def _recall(self, args: str = "") -> CommandResult:
        memory = self._memory_or_none()
        if memory is None:
            return CommandResult(output="Memory is not enabled.")

        query = args.strip()
        if not query:
            return CommandResult(output="Usage: /recall <texto a buscar>")

        facts, messages = memory.recall(query, all_sessions=True)

        if not facts and not messages:
            return CommandResult(output=f"Nada encontrado para: {query}")

        lines: list[str] = []
        if facts:
            lines.append("Facts:")
            for key, value in facts.items():
                lines.append(f"  {key} = {value}")

        if messages:
            if lines:
                lines.append("")
            lines.append("Messages:")
            for hit in messages:
                lines.append(f"  [{hit.role} @{hit.session_id}] {hit.content}")

        return CommandResult(output="\n".join(lines))

    def _export(self, args: str = "") -> CommandResult:
        memory = self._memory_or_none()
        if memory is None:
            return CommandResult(output="Memory is not enabled.")

        raw = args.strip()
        session = None
        path = ""
        if raw:
            parts = raw.split(maxsplit=1)
            if parts[0] == "session" and len(parts) > 1:
                session = parts[1].strip()
                path = ""
            else:
                path = raw

        if not path:
            target = session or memory.session_id
            path = f"liz-{target}.json"

        target = Path(path)
        if target.parent and not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)

        data = memory.export_session(session)
        target.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return CommandResult(
            output=f"Exported {len(data['messages'])} messages "
                   f"and {len(data['facts'])} facts to {target}"
        )

    def _import(self, args: str = "") -> CommandResult:
        memory = self._memory_or_none()
        if memory is None:
            return CommandResult(output="Memory is not enabled.")

        path = args.strip()
        if not path:
            return CommandResult(output="Usage: /import <archivo.json>")

        target = Path(path)
        if not target.exists():
            return CommandResult(output=f"File not found: {path}")

        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            return CommandResult(output=f"Invalid JSON: {error}")

        try:
            count = memory.import_session(data)
        except ValueError as error:
            return CommandResult(output=str(error))

        return CommandResult(
            output=f"Imported {count} messages into session "
                   f"'{memory.session_id}'"
        )

    def _voice(self, args: str = "") -> CommandResult:
        if self.voice is None:
            return CommandResult(output="Voice support is not enabled.")

        voice = self.voice
        action = args.strip().lower()

        if action == "on":
            if not voice.stt.available():
                return CommandResult(output=self._stt_install_help())
            if voice.running:
                return CommandResult(
                    output="The voice listener is already running."
                )
            voice.start()
            return CommandResult(
                output=f"Voice on. Say {voice.wake_words[0]!r} to talk to "
                "Liz; /voice off stops it."
            )

        if action == "off":
            was_running = voice.running
            voice.stop()
            return CommandResult(
                output="Voice off."
                if was_running
                else "The voice listener was not running."
            )

        if action:
            return CommandResult(output="Usage: /voice [on|off]")

        return CommandResult(output=self._voice_status())

    def _voice_status(self) -> str:
        voice = self.voice
        settings = self.settings

        stt_detail = voice.stt.name if voice.stt.available() else "not installed"

        engines = getattr(voice.tts, "engines", None) or [voice.tts]
        tts_detail = ", ".join(
            engine.name for engine in engines if engine.available()
        ) or "not installed"

        lines = [
            "Voice",
            f"  STT:        {stt_detail}",
            f"  TTS:        {tts_detail}",
            f"  Language:   {settings.voice_language}",
            f"  Record:     {settings.voice_record_seconds}s per window",
            "  Wake words: " + ", ".join(repr(w) for w in voice.wake_words),
            f"  Listener:   {'running' if voice.running else 'stopped'}",
            "",
            "  /talk           one voice turn",
            "  /voice on|off    background listener",
        ]
        if not voice.stt.available():
            lines += ["", self._stt_install_help()]
        return "\n".join(lines)

    @staticmethod
    def _stt_install_help() -> str:
        return (
            "No speech-to-text engine yet. Install whisper.cpp:\n"
            "  git clone https://github.com/ggml-org/whisper.cpp\n"
            "  cmake -B whisper.cpp/build -S whisper.cpp\n"
            "  cmake --build whisper.cpp/build -j\n"
            "  bash whisper.cpp/models/download-ggml-model.sh base\n"
            "Liz finds both automatically under whisper.cpp/, or you can "
            "point WHISPER_BIN and WHISPER_MODEL at them in .env."
        )

    def _talk(self, args: str = "") -> CommandResult:
        if self.voice is None:
            return CommandResult(output="Voice support is not enabled.")
        if not self.voice.stt.available():
            return CommandResult(output=self._stt_install_help())

        try:
            turn = self.voice.step(require_wake_word=False)
        except VoiceError as error:
            return CommandResult(output=f"Voice error: {error}")

        if turn is None:
            return CommandResult(output="No voice detected in the recording.")

        return CommandResult(
            output=f'You said: "{turn.transcript}"\nLiz: {turn.reply}'
        )

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
        import importlib

        for module in ("ddgs", "duckduckgo_search"):
            try:
                importlib.import_module(module)
            except ImportError:
                continue
            return ("WebSearch", "ok", f"{module} installed")
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
