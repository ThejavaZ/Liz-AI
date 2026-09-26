from liz.config.settings import Settings
from liz.core.context import Context
from liz.infrastructure.ai.base import AIProvider
from liz.infrastructure.ai.factory import create_provider


class Agent:
    def __init__(self, settings: Settings, system_prompt: str) -> None:
        self.settings = settings
        self.provider: AIProvider = create_provider(settings)
        self.context = Context(system_prompt)

    def chat(self, user_input: str) -> str:
        self.context.add_user_message(user_input)

        response = self.provider.chat(
            messages=self.context.get_messages(),
        )

        self.context.add_assistant_message(response.content)
        return response.content
