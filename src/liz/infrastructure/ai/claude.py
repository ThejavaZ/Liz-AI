import json
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
        claude_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                system_instruction = msg.content
                continue

            role: str
            blocks: list[dict[str, Any]] = []

            if msg.role == Role.USER:
                role = "user"
                blocks.append({"type": "text", "text": msg.content})

            elif msg.role == Role.ASSISTANT:
                role = "assistant"
                # Text must precede tool_use blocks in the content list.
                if msg.content:
                    blocks.append({"type": "text", "text": msg.content})
                for call in msg.tool_calls:
                    blocks.append({
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments
                        if isinstance(call.arguments, dict) else {},
                    })
                if not blocks:
                    # The API rejects an empty assistant message.
                    blocks.append({"type": "text", "text": "(no content)"})

            elif msg.role == Role.TOOL:
                role = "user"
                blocks.append({
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id or "",
                    # The API rejects empty content in a tool_result.
                    "content": msg.content or "(no output)",
                })

            else:
                continue

            # The API requires strictly alternating roles, and the agent
            # emits one Message per tool result, so consecutive messages
            # of the same role are merged into a single entry.
            if claude_messages and claude_messages[-1]["role"] == role:
                claude_messages[-1]["content"].extend(blocks)
            else:
                claude_messages.append({"role": role, "content": blocks})

        return system_instruction, claude_messages

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Anthropic names the JSON schema "input_schema".
        return [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get(
                    "parameters", {"type": "object", "properties": {}}
                ),
            }
            for tool in tools
        ]

    def _build_kwargs(
        self,
        system_instruction: str | None,
        claude_messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "messages": claude_messages,
        }
        if system_instruction:
            kwargs["system"] = system_instruction
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
        return kwargs

    @staticmethod
    def _parse_tool_calls(content: Any) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        for block in content or []:
            if getattr(block, "type", "") != "tool_use":
                continue
            raw_input = getattr(block, "input", None)
            arguments = raw_input if isinstance(raw_input, dict) else {}
            tool_calls.append(ToolCall(
                id=getattr(block, "id", "") or f"call_{len(tool_calls)}",
                name=getattr(block, "name", "") or "",
                arguments=arguments,
            ))
        return tool_calls

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> AIResponse:
        system_instruction, claude_messages = self._convert_messages(messages)
        kwargs = self._build_kwargs(system_instruction, claude_messages, tools)

        response = self.client.messages.create(**kwargs)

        text = "".join(
            block.text for block in response.content
            if getattr(block, "type", "") == "text"
            and isinstance(getattr(block, "text", None), str)
        )

        return AIResponse(
            content=text,
            model=response.model,
            tool_calls=self._parse_tool_calls(response.content),
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
        kwargs = self._build_kwargs(system_instruction, claude_messages, tools)

        text_parts: list[str] = []
        usage: dict = {}
        model = self.model
        accumulated: dict[int, dict[str, Any]] = {}

        with self.client.messages.stream(**kwargs) as stream:
            for event in stream:
                event_type = getattr(event, "type", "")

                if event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if block is not None and getattr(
                        block, "type", ""
                    ) == "tool_use":
                        index = getattr(event, "index", 0)
                        accumulated[index] = {
                            "id": getattr(block, "id", "") or "",
                            "name": getattr(block, "name", "") or "",
                            "arguments": "",
                        }

                elif event_type == "content_block_delta":
                    delta = event.delta
                    delta_type = getattr(delta, "type", "")

                    if delta_type == "text_delta":
                        text_parts.append(delta.text)
                        yield StreamChunk(text=delta.text)

                    elif delta_type == "input_json_delta":
                        index = getattr(event, "index", 0)
                        if index in accumulated:
                            accumulated[index][
                                "arguments"
                            ] += delta.partial_json

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
            if not isinstance(arguments, dict):
                arguments = {}

            tool_calls.append(ToolCall(
                id=entry["id"] or f"call_{index}",
                name=entry["name"],
                arguments=arguments,
            ))
        return tool_calls
