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

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        return AIResponse(
            content=response.text,
            model=self.model,
            usage={
                "input_tokens": response.usage_metadata.prompt_token_count or 0,
                "output_tokens": response.usage_metadata.candidates_token_count or 0,
            },
        )
