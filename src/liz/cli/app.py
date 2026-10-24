import sys
from pathlib import Path

from liz.config.settings import Settings
from liz.core.agent import Agent
from liz.core.memory import ConversationMemory
from liz.infrastructure.persistence.memory_store import SQLiteMemoryStore
from liz.tools.registry import ToolRegistry
from liz.tools.permissions import PermissionLayer, Permission
from liz.tools.filesystem import ListDirectory, ReadFile, FileExists
from liz.tools.git import GitStatus, GitLog, GitBranch
from liz.tools.terminal import RunCommand
from liz.cli.commands import CommandHandler


def load_system_prompt() -> str:
    prompt_path = "prompts/liz.txt"
    try:
        with open(prompt_path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "You are Liz, a helpful AI assistant."


def create_tool_registry() -> ToolRegistry:
    permissions = PermissionLayer()
    permissions.set_default(Permission.ALLOW)

    registry = ToolRegistry(permissions=permissions)
    registry.register(ListDirectory())
    registry.register(ReadFile())
    registry.register(FileExists())
    registry.register(GitStatus())
    registry.register(GitLog())
    registry.register(GitBranch())
    registry.register(RunCommand())

    return registry


def create_memory(settings: Settings) -> ConversationMemory:
    db_path = Path(settings.data_dir) / "liz.db"
    store = SQLiteMemoryStore(db_path)
    return ConversationMemory(store)


def run_cli() -> None:
    settings = Settings()
    system_prompt = load_system_prompt()
    tool_registry = create_tool_registry()
    memory = create_memory(settings)
    agent = Agent(
        settings,
        system_prompt,
        tool_registry=tool_registry,
        memory=memory,
    )
    commands = CommandHandler(
        settings,
        agent.context,
        tool_registry,
        provider=agent.provider,
        agent=agent,
    )

    model = getattr(agent.provider, "active", agent.provider).model
    print(f"Liz AI ({settings.ai_provider}/{model})")
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
                response = agent.chat(user_input)
                print(f"\nLiz: {response}\n")
            except Exception as e:
                print(f"\nError: {e}\n")
    finally:
        memory.close()


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
