from dataclasses import dataclass, field
from enum import Enum


class ToolStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    DENIED = "denied"


@dataclass
class ToolResult:
    status: ToolStatus
    data: str
    error: str | None = None

    @classmethod
    def success(cls, data: str) -> "ToolResult":
        return cls(status=ToolStatus.SUCCESS, data=data, error=None)

    @classmethod
    def error(cls, msg: str) -> "ToolResult":
        return cls(status=ToolStatus.ERROR, data="", error=msg)

    @classmethod
    def denied(cls, reason: str) -> "ToolResult":
        return cls(status=ToolStatus.DENIED, data="", error=reason)
