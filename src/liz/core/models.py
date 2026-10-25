from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """A single conversation turn.

    Providers that support tool calling need the assistant's requested
    calls and the matching tool response id, so both are carried on the
    message itself.
    """

    role: Role
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None


@dataclass
class ToolResultMessage:
    tool_call_id: str
    content: str


@dataclass
class AIResponse:
    content: str
    model: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


@dataclass
class StreamChunk:
    """A single piece of a streamed response.

    `text` carries incremental content. Tool calls are only present on
    the final chunk, once the provider has finished collecting them.
    `tool_events` describes tool activity between streaming rounds.
    """

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_events: list[str] = field(default_factory=list)
    done: bool = False
    response: AIResponse | None = None
