import sys

from liz.config.settings import Settings
from liz.core.agent import Agent


def load_system_prompt() -> str:
    prompt_path = "prompts/liz.txt"
    try:
        with open(prompt_path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "You are Liz, a helpful AI assistant."


def run_cli() -> None:
    settings = Settings()
    system_prompt = load_system_prompt()
    agent = Agent(settings, system_prompt)

    print(f"Liz AI ({settings.ai_provider}/{settings.ai_model})")
    print("Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        try:
            response = agent.chat(user_input)
            print(f"\nLiz: {response}\n")
        except Exception as e:
            print(f"\nError: {e}\n")


def main() -> None:
    run_cli()


if __name__ == "__main__":
    main()
