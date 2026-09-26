from abc import ABC, abstractmethod

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
    def requires_confirmation(self) -> bool:
        return False

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        pass
