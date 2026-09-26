from abc import ABC, abstractmethod
from typing import Any

from liz.core.models import AIResponse, Message


class AIProvider(ABC):

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> AIResponse:
        pass
