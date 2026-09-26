from abc import ABC, abstractmethod
from typing import Any

from liz.tools.models import ToolResult


class Tool(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    @property
    def requires_confirmation(self) -> bool:
        return False

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass
