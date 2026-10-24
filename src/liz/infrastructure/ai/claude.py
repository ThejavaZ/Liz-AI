from typing import Any, Iterator

import anthropic

from liz.core.models import (
    AIResponse,
    Message,
    Role,
    StreamChunk,
    ToolCall,
)
from liz.infrastructure.ai.base import AIProvider


class ClaudeProvider(AIProvider):
    def __init__(self, api_key: str | None, model: str) -> None:
        if not api_key:
            raise ValueError("Anthropic API key is required")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def _convert_messages(
        self, messages: list[Message]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        system_instruction = None
        claude_messages = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_instruction = msg.content
            elif msg.role == Role.USER:
                claude_messages.append(
                    {"role": "user", "content": msg.content}
                )
            elif msg.role == Role.ASSISTANT:
                claude_messages.append(
                    {"role": "assistant", "content": msg.content}
                )

        return system_instruction, claude_messages

    def _build_kwargs(
        self,
        system_instruction: str | None,
        claude_messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": claude_messages,
        }
        if system_instruction:
            kwargs["system"] = system_instruction
        return kwargs

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> AIResponse:
        system_instruction, claude_messages = self._convert_messages(messages)
        kwargs = self._build_kwargs(system_instruction, claude_messages)

        response = self.client.messages.create(**kwargs)

        text = "".join(
            block.text for block in response.content
            if isinstance(getattr(block, "text", None), str)
        )

        return AIResponse(
            content=text,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    def stream_chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[StreamChunk]:
        system_instruction, claude_messages = self._convert_messages(messages)
        kwargs = self._build_kwargs(system_instruction, claude_messages)

        text_parts: list[str] = []
        usage: dict = {}
        model = self.model

        with self.client.messages.stream(**kwargs) as stream:
            for event in stream:
                event_type = getattr(event, "type", "")

                if event_type == "content_block_delta":
                    delta = event.delta
                    if getattr(delta, "type", "") == "text_delta":
                        text_parts.append(delta.text)
                        yield StreamChunk(text=delta.text)

                elif event_type == "message_delta":
                    usage_update = getattr(event, "usage", None)
                    if usage_update is not None:
                        usage["output_tokens"] = getattr(
                            usage_update, "output_tokens", 0
                        )

                elif event_type == "message_start":
                    message = getattr(event, "message", None)
                    if message is not None:
                        model = getattr(message, "model", model)
                        usage_update = getattr(message, "usage", None)
                        if usage_update is not None:
                            usage["input_tokens"] = getattr(
                                usage_update, "input_tokens", 0
                            )

        response = AIResponse(
            content="".join(text_parts),
            model=model,
            usage=usage,
        )
        yield StreamChunk(
            text="",
            done=True,
            response=response,
        )
