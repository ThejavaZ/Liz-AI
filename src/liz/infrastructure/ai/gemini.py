import json
from typing import Any, Iterator

from google import genai
from google.genai import types

from liz.core.models import (
    AIResponse,
    Message,
    Role,
    StreamChunk,
    ToolCall,
)
from liz.infrastructure.ai.base import AIProvider


class GeminiProvider(AIProvider):
    def __init__(self, api_key: str | None, model: str) -> None:
        if not api_key:
            raise ValueError("Gemini API key is required")

        self.client = genai.Client(api_key=api_key)
        self.model = model
        self._call_seq = 0

    def _next_call_id(self, name: str) -> str:
        # Gemini assigns no id to function calls, so one is synthesized.
        # The counter keeps parallel calls to the same tool apart and the
        # name keeps ids distinct even across provider fallbacks.
        self._call_seq += 1
        return f"call_{self._call_seq}_{name}"

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

    def _extract_text(self, response: Any) -> str:
        text_parts = []
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "text") and part.text:
                    text_parts.append(part.text)
        return "".join(text_parts)

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[types.Content]]:
        system_instruction = None
        contents: list[types.Content] = []
        # Tool results carry only their id; remember which function each
        # id belongs to so the response part can use the real name
        # (Gemini matches it against the function_call in the history).
        call_names: dict[str, str] = {}

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_instruction = msg.content

            elif msg.role == Role.USER:
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=msg.content)],
                ))

            elif msg.role == Role.ASSISTANT:
                parts: list[types.Part] = []
                # An empty text part would round-trip as None and break
                # the next send_message, so only emit text when present.
                if msg.content:
                    parts.append(types.Part.from_text(text=msg.content))
                for call in msg.tool_calls:
                    call_names[call.id] = call.name
                    args = call.arguments if isinstance(
                        call.arguments, dict
                    ) else {}
                    parts.append(types.Part.from_function_call(
                        name=call.name,
                        args=args,
                    ))
                if not parts:
                    parts.append(types.Part.from_text(text="(no content)"))
                contents.append(types.Content(role="model", parts=parts))

            elif msg.role == Role.TOOL:
                name = call_names.get(msg.tool_call_id or "", "tool_result")
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_function_response(
                        name=name,
                        response={"result": msg.content},
                    )],
                ))

        return system_instruction, contents

    def _build_config(
        self,
        system_instruction: str | None,
        tools: list[dict[str, Any]] | None,
    ) -> types.GenerateContentConfig | None:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
        ) if system_instruction else None

        if tools:
            config = config or types.GenerateContentConfig()
            config.tools = self._convert_tools(tools)

        return config

    def _extract_tool_calls(self, response: Any) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                function_call = getattr(part, "function_call", None)
                if function_call:
                    args = function_call.args
                    tool_calls.append(ToolCall(
                        id=self._next_call_id(function_call.name),
                        name=function_call.name,
                        arguments=dict(args) if isinstance(args, dict) else {},
                    ))
        return tool_calls

    def _open_chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
    ):
        system_instruction, contents = self._convert_messages(messages)
        config = self._build_config(system_instruction, tools)

        chat = self.client.chats.create(
            model=self.model,
            config=config,
            history=contents[:-1] if len(contents) > 1 else [],
        )

        # send_message takes str | Part | list[Part]. The last entry may
        # be a function_response (after a tool round) whose text is None,
        # so the parts themselves are forwarded.
        if contents:
            last_message: Any = contents[-1].parts
        else:
            last_message = ""
        return chat, last_message

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> AIResponse:
        chat, last_message = self._open_chat(messages, tools)
        response = chat.send_message(last_message)

        return AIResponse(
            content=self._extract_text(response),
            model=self.model,
            tool_calls=self._extract_tool_calls(response),
            usage={
                "input_tokens": response.usage_metadata.prompt_token_count or 0,
                "output_tokens": response.usage_metadata.candidates_token_count or 0,
            },
        )

    def stream_chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[StreamChunk]:
        chat, last_message = self._open_chat(messages, tools)

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        usage: dict = {}
        # Streaming can repeat a function_call across chunks; ids are
        # unique per extraction, so dedup by name + arguments instead.
        seen: set[tuple[str, str]] = set()

        for chunk in chat.send_message_stream(last_message):
            piece = self._extract_text(chunk)
            if piece:
                text_parts.append(piece)
                yield StreamChunk(text=piece)

            for call in self._extract_tool_calls(chunk):
                key = (call.name, json.dumps(call.arguments, sort_keys=True))
                if key in seen:
                    continue
                seen.add(key)
                tool_calls.append(call)

            metadata = getattr(chunk, "usage_metadata", None)
            if metadata is not None:
                usage = {
                    "input_tokens": metadata.prompt_token_count or 0,
                    "output_tokens": metadata.candidates_token_count or 0,
                }

        response = AIResponse(
            content="".join(text_parts),
            model=self.model,
            tool_calls=tool_calls,
            usage=usage,
        )
        yield StreamChunk(
            text="",
            tool_calls=tool_calls,
            done=True,
            response=response,
        )
