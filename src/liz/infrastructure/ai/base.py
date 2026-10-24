from abc import ABC, abstractmethod
from typing import Any, Iterator

from liz.core.models import AIResponse, Message, StreamChunk


class AIProvider(ABC):

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> AIResponse:
        pass

    def stream_chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
    ) -> Iterator[StreamChunk]:
        """Stream a response.

        Providers that support real streaming override this. The default
        implementation degrades to a single chunk so callers never have
        to check whether streaming is available.
        """
        response = self.chat(messages, tools)
        yield StreamChunk(
            text=response.content,
            tool_calls=response.tool_calls,
            done=True,
            response=response,
        )
