from typing import Any

from google import genai
from google.genai import types

from liz.core.models import AIResponse, Message, Role, ToolCall
from liz.infrastructure.ai.base import AIProvider


class GeminiProvider(AIProvider):
    def __init__(self, api_key: str | None, model: str) -> None:
        if not api_key:
            raise ValueError("Gemini API key is required")

        self.client = genai.Client(api_key=api_key)
        self.model = model

    def _convert_tools(
        self, tools: list[dict[str, Any]]
    ) -> list[types.Tool]:
        gemini_tools = []
        for tool in tools:
            function = types.FunctionDeclaration(
                name=tool["name"],
                description=tool["description"],
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        k: self._convert_schema_type(v)
                        for k, v in tool.get("parameters", {})
                            .get("properties", {}).items()
                    },
                    required=tool.get("parameters", {}).get("required", []),
                ),
            )
            gemini_tools.append(types.Tool(function_declarations=[function]))
        return gemini_tools

    def _convert_schema_type(self, prop: dict[str, Any]) -> types.Schema:
        type_map = {
            "string": types.Type.STRING,
            "integer": types.Type.INTEGER,
            "number": types.Type.NUMBER,
            "boolean": types.Type.BOOLEAN,
        }
        return types.Schema(
            type=type_map.get(prop.get("type", "string"), types.Type.STRING),
            description=prop.get("description"),
        )

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
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

        if tools:
            config = config or types.GenerateContentConfig()
            config.tools = self._convert_tools(tools)

        chat = self.client.chats.create(
            model=self.model,
            config=config,
            history=contents[:-1] if len(contents) > 1 else [],
        )

        last_message = contents[-1].parts[0].text if contents else ""
        response = chat.send_message(last_message)

        tool_calls = []
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    tool_calls.append(ToolCall(
                        id=f"call_{part.function_call.name}",
                        name=part.function_call.name,
                        arguments=dict(part.function_call.args),
                    ))

        return AIResponse(
            content=response.text or "",
            model=self.model,
            tool_calls=tool_calls,
            usage={
                "input_tokens": response.usage_metadata.prompt_token_count or 0,
                "output_tokens": response.usage_metadata.candidates_token_count or 0,
            },
        )
