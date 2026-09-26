from openai import OpenAI

from liz.core.models import AIResponse, Message, Role
from liz.infrastructure.ai.base import AIProvider


class DeepSeekProvider(AIProvider):
    BASE_URL = "https://api.deepseek.com"

    def __init__(self, api_key: str | None, model: str) -> None:
        if not api_key:
            raise ValueError("DeepSeek API key is required")

        self.client = OpenAI(api_key=api_key, base_url=self.BASE_URL)
        self.model = model

    def chat(
        self,
        messages: list[Message],
    ) -> AIResponse:
        ds_messages = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                ds_messages.append({"role": "system", "content": msg.content})
            elif msg.role == Role.USER:
                ds_messages.append({"role": "user", "content": msg.content})
            elif msg.role == Role.ASSISTANT:
                ds_messages.append({"role": "assistant", "content": msg.content})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=ds_messages,
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
