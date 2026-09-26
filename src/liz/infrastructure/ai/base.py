from abc import ABC, abstractmethod

from liz.core.models import AIResponse


class AIProvider(ABC):

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> AIResponse:
        pass