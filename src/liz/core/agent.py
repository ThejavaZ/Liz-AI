from typing import Iterator

from liz.config.settings import Settings
from liz.core.context import Context
from liz.core.memory import ConversationMemory
from liz.core.models import AIResponse, Message, Role, StreamChunk, ToolCall
from liz.core.summarizer import ContextSummarizer
from liz.infrastructure.ai.base import AIProvider
from liz.infrastructure.ai.factory import create_provider
from liz.tools.registry import ToolRegistry
from liz.tools.models import ToolStatus

MAX_TOOL_ROUNDS = 5


def _short_args(tool_call: ToolCall, limit: int = 40) -> str:
    text = ", ".join(f"{k}={v!r}" for k, v in tool_call.arguments.items())
    return text if len(text) <= limit else text[: limit - 3] + "..."


class Agent:
    def __init__(
        self,
        settings: Settings,
        system_prompt: str,
        tool_registry: ToolRegistry | None = None,
        memory: ConversationMemory | None = None,
    ) -> None:
        self.settings = settings
        self.provider: AIProvider = create_provider(settings)
        self.context = Context(system_prompt)
        self.tool_registry = tool_registry
        self.memory = memory
        self.summarizer = ContextSummarizer(self.provider)

        if memory is not None:
            self._restore_history()

    def _restore_history(self) -> None:
        for message in self.memory.load_history():
            self.context.messages.append(message)

    def _condense(self) -> None:
        """Shrink the context before it grows past the threshold."""
        if self.summarizer.should_summarize(self.context):
            self.summarizer.summarize(self.context)

    def switch_session(self, session_id: str) -> str:
        """Move the conversation to a named session."""
        if self.memory is None:
            return session_id

        self.memory.switch(session_id)
        self.context.clear()
        self._restore_history()
        return session_id

    def chat(self, user_input: str) -> str:
        self.context.add_user_message(user_input)
        if self.memory is not None:
            self.memory.remember(Message(role=Role.USER, content=user_input))

        tools = self.tool_registry.get_tool_schemas() if self.tool_registry else None
        self._condense()

        for _ in range(MAX_TOOL_ROUNDS):
            response = self.provider.chat(
                messages=self.context.get_messages(),
                tools=tools,
            )

            if not response.tool_calls:
                self.context.add_assistant_message(response.content)
                if self.memory is not None:
                    self.memory.remember(
                        Message(role=Role.ASSISTANT, content=response.content)
                    )
                return response.content

            self.context.add_assistant_message(
                response.content or "", tool_calls=response.tool_calls
            )

            for tool_call in response.tool_calls:
                result = self._execute_tool(tool_call)
                self._add_tool_result(tool_call, result)

        final_response = self.provider.chat(
            messages=self.context.get_messages(),
            tools=tools,
        )
        self.context.add_assistant_message(final_response.content)
        if self.memory is not None:
            self.memory.remember(
                Message(role=Role.ASSISTANT, content=final_response.content)
            )
        return final_response.content

    def clear_conversation(self) -> None:
        self.context.clear()
        if self.memory is not None:
            self.memory.clear()

    def stream_chat(self, user_input: str) -> Iterator[StreamChunk]:
        """Stream a reply, executing tools between rounds when needed.

        Text chunks are forwarded as they arrive. When the model asks
        for tools, the results are fed back and a new round is streamed;
        tool activity is reported through `tool_events`.
        """
        self.context.add_user_message(user_input)
        if self.memory is not None:
            self.memory.remember(Message(role=Role.USER, content=user_input))

        tools = self.tool_registry.get_tool_schemas() if self.tool_registry else None
        self._condense()

        for _ in range(MAX_TOOL_ROUNDS):
            text_parts: list[str] = []
            final: AIResponse | None = None

            for chunk in self.provider.stream_chat(
                messages=self.context.get_messages(),
                tools=tools,
            ):
                if chunk.text:
                    text_parts.append(chunk.text)
                if chunk.done:
                    final = chunk.response
                yield chunk

            if final is None:
                final = AIResponse(
                    content="".join(text_parts),
                    model=self.provider.model
                    if hasattr(self.provider, "model") else "",
                    tool_calls=[],
                )

            if not final.tool_calls:
                self.context.add_assistant_message(final.content)
                if self.memory is not None:
                    self.memory.remember(
                        Message(role=Role.ASSISTANT, content=final.content)
                    )
                return

            self.context.add_assistant_message(
                final.content or "", tool_calls=final.tool_calls
            )

            events: list[str] = []
            for tool_call in final.tool_calls:
                result = self._execute_tool(tool_call)
                events.append(f"{tool_call.name}({_short_args(tool_call)})")
                self._add_tool_result(tool_call, result)

            yield StreamChunk(tool_events=events)

        final_response = self.provider.chat(
            messages=self.context.get_messages(),
            tools=tools,
        )
        self.context.add_assistant_message(final_response.content)
        if self.memory is not None:
            self.memory.remember(
                Message(role=Role.ASSISTANT, content=final_response.content)
            )
        yield StreamChunk(
            text=final_response.content,
            done=True,
            response=final_response,
        )

    def _execute_tool(self, tool_call: ToolCall) -> str:
        if self.tool_registry is None:
            return "Error: No tool registry configured"

        result = self.tool_registry.execute(
            tool_call.name, **tool_call.arguments
        )

        if result.status == ToolStatus.SUCCESS:
            return result.data
        elif result.status == ToolStatus.DENIED:
            return f"Permission denied: {result.error}"
        else:
            return f"Error: {result.error}"

    def _add_tool_result(self, tool_call: ToolCall, result: str) -> None:
        self.context.add_tool_result(tool_call.id, result)
