from dataclasses import dataclass, field
from enum import Enum


class Role(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    role: Role
    content: str


@dataclass
class AIResponse:
    content: str
    model: str
    usage: dict = field(default_factory=dict)
