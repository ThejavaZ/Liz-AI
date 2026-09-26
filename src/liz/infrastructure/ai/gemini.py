import google.generativeai as genai

from liz.core.models import AIResponse, Message, Role
from liz.infrastructure.ai.base import AIProvider


class GeminiProvider(AIProvider):
    def __init__(self, api_key: str | None, model: str) -> None:
        if not api_key:
            raise ValueError("Gemini API key is required")

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)

    def chat(
        self,
        messages: list[Message],
    ) -> AIResponse:
        history = []
        system_instruction = None

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_instruction = msg.content
            elif msg.role == Role.USER:
                history.append({"role": "user", "parts": [msg.content]})
            elif msg.role == Role.ASSISTANT:
                history.append({"role": "model", "parts": [msg.content]})

        if system_instruction:
            self.model = genai.GenerativeModel(
                self.model.model_name,
                system_instruction=system_instruction,
            )

        chat = self.model.start_chat(history=history[:-1] if history else [])

        last_user_msg = history[-1]["parts"][0] if history else ""
        response = chat.send_message(last_user_msg)

        return AIResponse(
            content=response.text,
            model=self.model.model_name,
            usage={
                "input_tokens": response.usage_metadata.prompt_token_count,
                "output_tokens": response.usage_metadata.candidates_token_count,
            },
        )
