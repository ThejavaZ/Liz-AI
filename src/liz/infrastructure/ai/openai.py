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
                oai_messages.append(
                    {"role": "assistant", "content": msg.content}
                )
        return oai_messages

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
        import json

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
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self._convert_messages(messages),
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

    def stream_chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[StreamChunk]:
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=self._convert_messages(messages),
            stream=True,
            stream_options={"include_usage": True},
        )

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
