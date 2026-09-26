from openai import OpenAI

from liz.core.models import AIResponse, Message, Role
from liz.infrastructure.ai.base import AIProvider


class OpenAIProvider(AIProvider):
    def __init__(self, api_key: str | None, model: str) -> None:
        if not api_key:
            raise ValueError("OpenAI API key is required")

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def chat(
        self,
        messages: list[Message],
    ) -> AIResponse:
        oai_messages = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                oai_messages.append({"role": "system", "content": msg.content})
            elif msg.role == Role.USER:
                oai_messages.append({"role": "user", "content": msg.content})
            elif msg.role == Role.ASSISTANT:
                oai_messages.append({"role": "assistant", "content": msg.content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=oai_messages,
        )

        choice = response.choices[0]

        return AIResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "input_tokens": response.usage.prompt_token_count,
                "output_tokens": response.usage.completion_token_count,
            } if response.usage else {},
        )
