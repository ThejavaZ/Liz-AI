from liz.config.settings import Settings
from liz.core.context import Context
from liz.core.models import ToolCall
from liz.infrastructure.ai.base import AIProvider
from liz.infrastructure.ai.factory import create_provider
from liz.tools.registry import ToolRegistry
from liz.tools.models import ToolStatus

MAX_TOOL_ROUNDS = 5


class Agent:
    def __init__(
        self,
        settings: Settings,
        system_prompt: str,
        tool_registry: ToolRegistry | None = None,
    ) -> None:
        self.settings = settings
        self.provider: AIProvider = create_provider(settings)
        self.context = Context(system_prompt)
        self.tool_registry = tool_registry

    def chat(self, user_input: str) -> str:
        self.context.add_user_message(user_input)

        tools = self.tool_registry.get_tool_schemas() if self.tool_registry else None

        for _ in range(MAX_TOOL_ROUNDS):
            response = self.provider.chat(
                messages=self.context.get_messages(),
                tools=tools,
            )

            if not response.tool_calls:
                self.context.add_assistant_message(response.content)
                return response.content

            self.context.add_assistant_message(response.content or "")

            for tool_call in response.tool_calls:
                result = self._execute_tool(tool_call)
                self._add_tool_result(tool_call, result)

        final_response = self.provider.chat(
            messages=self.context.get_messages(),
            tools=tools,
        )
        self.context.add_assistant_message(final_response.content)
        return final_response.content

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
        from liz.core.models import Message, Role
        self.context.messages.append(
            Message(role=Role.TOOL, content=result)
        )
