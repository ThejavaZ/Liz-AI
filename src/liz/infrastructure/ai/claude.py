import anthropic

from liz.core.models import AIResponse, Message, Role
from liz.infrastructure.ai.base import AIProvider


class ClaudeProvider(AIProvider):
    def __init__(self, api_key: str | None, model: str) -> None:
        if not api_key:
            raise ValueError("Anthropic API key is required")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def chat(
        self,
        messages: list[Message],
    ) -> AIResponse:
        system_instruction = None
        claude_messages = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_instruction = msg.content
            elif msg.role == Role.USER:
                claude_messages.append({"role": "user", "content": msg.content})
            elif msg.role == Role.ASSISTANT:
                claude_messages.append({"role": "assistant", "content": msg.content})

        kwargs = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": claude_messages,
        }
        if system_instruction:
            kwargs["system"] = system_instruction

        response = self.client.messages.create(**kwargs)

        return AIResponse(
            content=response.content[0].text,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )
