import sys
from pathlib import Path

from liz.config.settings import Settings
from liz.core.agent import Agent
from liz.core.memory import ConversationMemory
from liz.infrastructure.persistence.memory_store import SQLiteMemoryStore
from liz.infrastructure.persistence.notes_store import SQLiteNotesStore
from liz.tools.registry import ToolRegistry
from liz.tools.permissions import PermissionLayer, Permission
from liz.tools.filesystem import ListDirectory, ReadFile, FileExists
from liz.tools.git import GitStatus, GitLog, GitBranch
from liz.tools.terminal import RunCommand
from liz.tools.calculator import Calculator
from liz.tools.web import WebSearch
from liz.tools.notes import Notes
from liz.tools.memory import Memory as MemoryTool
from liz.cli.commands import CommandHandler
from liz.cli.markdown import MarkdownRenderer
from liz.cli.voice import create_voice_loop

# Read-only or self-contained tools: never prompt the user for these.
ALWAYS_ALLOWED_TOOLS = (
    "list_directory",
    "read_file",
    "file_exists",
    "git_status",
    "git_log",
    "git_branch",
    "calculator",
    "web_search",
    "notes",
    "memory",
)

# Everything else (unknown future tools included) falls back to ASK.
ASK_RULES = (
    ("run_command", None),
    ("notes", "delete"),
    ("memory", "forget"),
)

_CONFIRM_ANSWERS = {"y", "yes", "s", "si", "sí"}


def load_system_prompt() -> str:
    prompt_path = "prompts/liz.txt"
    try:
        with open(prompt_path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "You are Liz, a helpful AI assistant."


def parse_fs_roots(settings: Settings) -> list[Path]:
    """Directories the tools may touch: FS_ROOTS, else home + cwd."""
    raw = settings.fs_roots.strip()
    if raw:
        return [
            Path(part.strip()).expanduser()
            for part in raw.split(",")
            if part.strip()
        ]
    return [Path.home(), Path.cwd()]


def confirm_tool_request(tool_name: str, preview: str) -> bool:
    """Interactive yes/no prompt for tools marked ASK."""
    detail = f" ({preview})" if preview else ""
    try:
        answer = input(f"\n  ⚠ Liz wants to use {tool_name}{detail}. "
                       "Allow? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    return answer.strip().lower() in _CONFIRM_ANSWERS


def create_tool_registry(
    notes_store=None,
    memory=None,
    roots: list[Path] | None = None,
    confirm=None,
) -> ToolRegistry:
    permissions = PermissionLayer()
    permissions.set_default(Permission.ASK)
    for tool_name in ALWAYS_ALLOWED_TOOLS:
        permissions.set_rule(tool_name, Permission.ALLOW)
    for tool_name, action in ASK_RULES:
        permissions.set_rule(tool_name, Permission.ASK, action=action)

    registry = ToolRegistry(permissions=permissions, confirm=confirm)
    registry.register(ListDirectory(roots))
    registry.register(ReadFile(roots))
    registry.register(FileExists(roots))
    registry.register(GitStatus())
    registry.register(GitLog())
    registry.register(GitBranch())
    registry.register(RunCommand(roots=roots))
    registry.register(Calculator())
    registry.register(WebSearch())

    if notes_store is not None:
        registry.register(Notes(notes_store))

    if memory is not None:
        registry.register(MemoryTool(memory))

    return registry


def create_memory(settings: Settings) -> ConversationMemory:
    db_path = Path(settings.data_dir) / "liz.db"
    store = SQLiteMemoryStore(db_path)
    return ConversationMemory(store)


def create_notes_store(settings: Settings) -> SQLiteNotesStore:
    db_path = Path(settings.data_dir) / "notes.db"
    return SQLiteNotesStore(db_path)


def warn_unknown_model(settings: Settings) -> None:
    """Warn when the configured model is not in the known list.

    An unknown model means every request pays for a failed call plus
    the fallback, which feels like high latency.
    """
    if not settings.ai_model:
        return

    from liz.infrastructure.ai.models_registry import list_models

    known = list_models(settings.ai_provider)
    if known and settings.ai_model not in known:
        print(
            f"Warning: model '{settings.ai_model}' is not a known "
            f"{settings.ai_provider} model.\n"
            f"         Every request will fail over to a fallback, which "
            f"adds latency.\n"
            f"         Fix AI_MODEL in .env or run /models to see options.\n"
        )


def _stream_response(agent, user_input: str) -> None:
    """Print the model's reply token by token as it arrives."""
    print("\nLiz: ", end="", flush=True)

    renderer = MarkdownRenderer()
    printed_any = False

    def _emit(text: str) -> None:
        nonlocal printed_any
        if text:
            print(text, end="", flush=True)
            printed_any = True

    for chunk in agent.stream_chat(user_input):
        if chunk.tool_events:
            # flush the pending line first so the tool activity does not
            # swallow part of the answer
            _emit(renderer.flush())
            if printed_any:
                print()
            for event in chunk.tool_events:
                print(f"  ⚙ {event}", flush=True)
            print("Liz: ", end="", flush=True)
            continue

        if chunk.text:
            _emit(renderer.feed(chunk.text))

    _emit(renderer.flush())
    print()


def run_cli() -> None:
    settings = Settings()
    system_prompt = load_system_prompt()
    notes_store = create_notes_store(settings)
    memory = create_memory(settings)
    tool_registry = create_tool_registry(
        notes_store,
        memory,
        roots=parse_fs_roots(settings),
        confirm=confirm_tool_request,
    )
    agent = Agent(
        settings,
        system_prompt,
        tool_registry=tool_registry,
        memory=memory,
    )
    voice = create_voice_loop(
        agent,
        settings,
        on_event=lambda message: print(message, flush=True),
    )
    commands = CommandHandler(
        settings,
        agent.context,
        tool_registry,
        provider=agent.provider,
        agent=agent,
        notes_store=notes_store,
        voice=voice,
    )

    model = getattr(agent.provider, "active", agent.provider).model
    print(f"Liz AI ({settings.ai_provider}/{model})")
    warn_unknown_model(settings)
    print("Type /help for commands, or /exit to leave.\n")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            result = commands.handle(user_input)
            if result is not None:
                print(f"\n{result.output}\n")
                if result.should_exit:
                    break
                continue

            if user_input.lower() in ("exit", "quit"):
                print("\nGoodbye!")
                break

            try:
                _stream_response(agent, user_input)
            except KeyboardInterrupt:
                print("\n[cancelled]")
            except Exception as e:
                print(f"\nError: {e}\n")
    finally:
        voice.stop()
        memory.close()
        notes_store.close()


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
