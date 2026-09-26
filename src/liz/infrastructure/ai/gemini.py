from google import genai
from google.genai import types

from liz.core.models import AIResponse, Message, Role
from liz.infrastructure.ai.base import AIProvider


class GeminiProvider(AIProvider):
    def __init__(self, api_key: str | None, model: str) -> None:
        if not api_key:
            raise ValueError("Gemini API key is required")

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def chat(
        self,
        messages: list[Message],
    ) -> AIResponse:
        system_instruction = None
        contents = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_instruction = msg.content
            elif msg.role == Role.USER:
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=msg.content)],
                ))
            elif msg.role == Role.ASSISTANT:
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=msg.content)],
                ))

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
        ) if system_instruction else None

        chat = self.client.chats.create(
            model=self.model,
            config=config,
            history=contents[:-1] if len(contents) > 1 else [],
        )

        last_message = contents[-1].parts[0].text if contents else ""
        response = chat.send_message(last_message)

        return AIResponse(
            content=response.text,
            model=self.model,
            usage={
                "input_tokens": response.usage_metadata.prompt_token_count or 0,
                "output_tokens": response.usage_metadata.candidates_token_count or 0,
            },
        )
