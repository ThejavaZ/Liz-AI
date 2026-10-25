"""Context summarization.

When the conversation outgrows a comfortable size, older turns are
collapsed into a short summary so the model keeps the gist without
paying for the whole transcript. If the model is unreachable the
summary falls back to plain truncation, so context always stays bounded.
"""

from __future__ import annotations

from liz.core.context import Context
from liz.core.models import Message, Role
from liz.infrastructure.ai.base import AIProvider

SUMMARY_PROMPT = (
    "Summarize the following conversation in under 150 words. "
    "Keep the user's goals, decisions and any facts that still matter. "
    "Answer with the summary only.\n\n{transcript}"
)

_NO_TOOL_PAIRS_MESSAGE = (
    "[Earlier conversation was summarized]\n{summary}"
)


class ContextSummarizer:
    def __init__(
        self,
        provider: AIProvider,
        threshold: int = 30,
        keep_recent: int = 10,
        max_chars: int = 4000,
    ) -> None:
        self.provider = provider
        self.threshold = threshold
        self.keep_recent = keep_recent
        self.max_chars = max_chars

    def should_summarize(self, context: Context) -> bool:
        return len(context.messages) > self.threshold

    def summarize(self, context: Context) -> bool:
        """Collapse older turns into one summary message.

        Returns True when the context was rewritten.
        """
        split = self._safe_split(context.messages)
        if split <= 1:
            return False

        older = context.messages[1:split]
        transcript = self._transcript(older)
        if not transcript:
            return False

        summary = self._summarize_text(transcript)

        placeholder = Message(
            role=Role.ASSISTANT,
            content=_NO_TOOL_PAIRS_MESSAGE.format(summary=summary),
        )
        context.messages = [context.messages[0], placeholder, *context.messages[split:]]
        return True

    def _safe_split(self, messages: list[Message]) -> int:
        """Pick the boundary between collapsed and kept messages.

        The kept range must start at a non-tool message and the collapsed
        range must not end on an assistant turn that requested tools,
        otherwise a result would be left without its call.
        """
        split = max(1, len(messages) - self.keep_recent)

        while split > 1 and (
            messages[split].role is Role.TOOL
            or messages[split - 1].tool_calls
        ):
            split -= 1

        return split

    @staticmethod
    def _transcript(messages: list[Message]) -> str:
        lines = []
        for message in messages:
            if message.role is Role.USER:
                lines.append(f"User: {message.content}")
            elif message.role is Role.ASSISTANT and message.content:
                lines.append(f"Liz: {message.content}")
        return "\n".join(lines)

    def _summarize_text(self, transcript: str) -> str:
        if len(transcript) > self.max_chars:
            transcript = transcript[-self.max_chars:]

        try:
            response = self.provider.chat(
                [Message(role=Role.USER, content=SUMMARY_PROMPT.format(
                    transcript=transcript
                ))]
            )
            summary = (response.content or "").strip()
            if summary:
                return summary
        except Exception:
            pass

        # Fallback: keep the tail of the transcript so context still shrinks.
        tail = transcript[-800:]
        return f"(summary unavailable) ...{tail}"
