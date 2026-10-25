import json
from typing import Any, Iterator

from openai import OpenAI

from liz.core.models import (
    AIResponse,
    Message,
    Role,
    StreamChunk,
    ToolCall,
)
from liz.infrastructure.ai.base import AIProvider


class OpenAIProvider(AIProvider):
    def __init__(
        self,
        api_key: str | None,
        model: str,
        base_url: str | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenAI API key is required")

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        self.client = OpenAI(**kwargs)
        self.model = model

    def _convert_messages(
        self, messages: list[Message]
    ) -> list[dict[str, Any]]:
        oai_messages = []
        for msg in messages:
            if msg.role == Role.SYSTEM:
                oai_messages.append({"role": "system", "content": msg.content})

            elif msg.role == Role.USER:
                oai_messages.append({"role": "user", "content": msg.content})

            elif msg.role == Role.ASSISTANT:
                entry: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.content or None,
                }
                if msg.tool_calls:
                    entry["tool_calls"] = [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments),
                            },
                        }
                        for call in msg.tool_calls
                    ]
                oai_messages.append(entry)

            elif msg.role == Role.TOOL:
                oai_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.tool_call_id or "",
                    "content": msg.content,
                })
        return oai_messages

    @staticmethod
    def _parse_tool_calls(message: Any) -> list[ToolCall]:
        raw = getattr(message, "tool_calls", None) or []
        tool_calls: list[ToolCall] = []

        for call in raw:
            function = getattr(call, "function", None)
            arguments: dict[str, Any] = {}
            raw_args = getattr(function, "arguments", None) if function else None
            if raw_args:
                try:
                    arguments = json.loads(raw_args)
                except json.JSONDecodeError:
                    arguments = {}

            tool_calls.append(ToolCall(
                id=getattr(call, "id", "") or f"call_{len(tool_calls)}",
                name=getattr(function, "name", "") if function else "",
                arguments=arguments,
            ))
        return tool_calls

    def _merge_tool_call(
        self, accumulated: dict[int, dict[str, Any]], parts: Any
    ) -> None:
        for part in parts or []:
            index = getattr(part, "index", 0)
            entry = accumulated.setdefault(
                index, {"id": "", "name": "", "arguments": ""}
            )
            if part.id:
                entry["id"] = part.id
            if part.function and part.function.name:
                entry["name"] = part.function.name
            if part.function and part.function.arguments:
                entry["arguments"] += part.function.arguments

    @staticmethod
    def _finalize_tool_calls(
        accumulated: dict[int, dict[str, Any]],
    ) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        for index in sorted(accumulated):
            entry = accumulated[index]
            raw = entry["arguments"] or "{}"
            try:
                arguments = json.loads(raw)
            except json.JSONDecodeError:
                arguments = {}

            tool_calls.append(ToolCall(
                id=entry["id"] or f"call_{index}",
                name=entry["name"],
                arguments=arguments,
            ))
        return tool_calls

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> AIResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_messages(messages),
        }
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        response = self.client.chat.completions.create(**kwargs)

        choice = response.choices[0]

        return AIResponse(
            content=choice.message.content or "",
            model=response.model,
            tool_calls=self._parse_tool_calls(choice.message),
            usage={
                "input_tokens": response.usage.prompt_token_count,
                "output_tokens": response.usage.completion_token_count,
            } if response.usage else {},
        )

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                },
            }
            for tool in tools
        ]

    def stream_chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[StreamChunk]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = self._convert_tools(tools)

        stream = self.client.chat.completions.create(**kwargs)

        text_parts: list[str] = []
        accumulated: dict[int, dict[str, Any]] = {}
        usage: dict = {}
        model = self.model

        for chunk in stream:
            if getattr(chunk, "model", None):
                model = chunk.model

            if getattr(chunk, "usage", None):
                usage = {
                    "input_tokens": chunk.usage.prompt_tokens or 0,
                    "output_tokens": chunk.usage.completion_tokens or 0,
                }

            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta

            if delta and delta.content:
                text_parts.append(delta.content)
                yield StreamChunk(text=delta.content)

            if delta and delta.tool_calls:
                self._merge_tool_call(accumulated, delta.tool_calls)

        tool_calls = self._finalize_tool_calls(accumulated)
        response = AIResponse(
            content="".join(text_parts),
            model=model,
            tool_calls=tool_calls,
            usage=usage,
        )
        yield StreamChunk(
            text="",
            tool_calls=tool_calls,
            done=True,
            response=response,
        )
