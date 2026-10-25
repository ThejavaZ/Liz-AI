"""Terminal rendering of the model's Markdown output.

Models answer in Markdown; a terminal cannot display it, so this renderer
translates the common subset (bold, italic, inline code, code fences,
headings, lists, quotes, links, rules and tables) into ANSI sequences
while the stream is still arriving.

It is standard library only and buffers text until a newline arrives, so
streaming still looks live. Output degrades to plain text when stdout is
not a TTY, when ``NO_COLOR`` is set or when ``TERM=dumb``.
"""

from __future__ import annotations

import os
import re
import sys

RESET = "\033[0m"
BOLD = "\033[1m"
ITALIC = "\033[3m"
DIM = "\033[2m"
UNDERLINE = "\033[4m"
CYAN = "\033[36m"
YELLOW = "\033[33m"

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST = re.compile(r"^(\s*)([-*+]|\d+\.)\s+(.*)$")
_QUOTE = re.compile(r"^\s*>\s?(.*)$")
_RULE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")
_FENCE = re.compile(r"^\s*```(\S*)\s*$")
_SEPARATOR = re.compile(r"^:?-{3,}:?$")

_CODE_SPAN = re.compile(r"`([^`]+)`")
_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_STRIKE = re.compile(r"~~(.+?)~~")
_ITALIC = re.compile(r"\*([^*\n]+)\*|(?<!\w)_([^_\n]+)_(?!\w)")

_RULE_WIDTH = 40


def supports_color(stream=None) -> bool:
    """Whether ANSI colour should be used for this stream."""
    stream = stream if stream is not None else sys.stdout
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("TERM", "") == "dumb":
        return False
    try:
        return bool(stream.isatty())
    except Exception:
        return False


def _inline(text: str, resume: str = "") -> str:
    """Render inline Markdown inside ``text``.

    ``resume`` re-opens an enclosing style after a reset, so a heading
    stays styled past a bold span or a code span.
    """
    parts = _CODE_SPAN.split(text)
    rendered: list[str] = []
    for index, part in enumerate(parts):
        if index % 2:
            rendered.append(f"{CYAN}{part}{RESET}{resume}")
        else:
            rendered.append(_styles(part, resume))
    return "".join(rendered)


def _styles(text: str, resume: str) -> str:
    def bold(match: re.Match) -> str:
        inner = match.group(1) or match.group(2)
        return f"{BOLD}{inner}{RESET}{resume}"

    def link(match: re.Match) -> str:
        label, url = match.groups()
        return f"{label} {DIM}({url}){RESET}{resume}"

    def strike(match: re.Match) -> str:
        return f"{DIM}{match.group(1)}{RESET}{resume}"

    def italic(match: re.Match) -> str:
        inner = match.group(1) or match.group(2)
        return f"{ITALIC}{inner}{RESET}{resume}"

    text = _LINK.sub(link, text)
    text = _BOLD.sub(bold, text)
    text = _STRIKE.sub(strike, text)
    text = _ITALIC.sub(italic, text)
    return text


class MarkdownRenderer:
    """Incrementally converts streamed Markdown into ANSI text."""

    def __init__(self, enabled: bool | None = None) -> None:
        self.enabled = supports_color() if enabled is None else enabled
        self._buffer = ""
        self._in_fence = False
        self._pending_table: list[str] = []

    def feed(self, text: str) -> str:
        """Consume a chunk and return everything safely printable now."""
        if not self.enabled:
            return text

        self._buffer += text
        if "\n" not in self._buffer:
            return ""

        *lines, self._buffer = self._buffer.split("\n")
        out = []
        for line in lines:
            rendered = self._render(line)
            if rendered is not None:
                out.append(f"{rendered}\n")
        return "".join(out)

    def flush(self) -> str:
        """Render the pending partial line (no trailing newline)."""
        if not self.enabled:
            return ""

        out = ""
        line, self._buffer = self._buffer, ""
        if line:
            rendered = self._render(line)
            if rendered is not None:
                out += rendered
        # a table row returns None and waits for its siblings
        return (out + self._drain_table()).rstrip("\n")

    @staticmethod
    def _is_table_row(line: str) -> bool:
        return line.lstrip().startswith("|") and line.rstrip().endswith("|")

    def _drain_table(self) -> str:
        """Render every buffered table row with consistent column widths."""
        if not self._pending_table:
            return ""
        rows = self._pending_table
        self._pending_table = []
        return self._render_table(rows)

    def _render(self, line: str) -> str | None:
        fence = _FENCE.match(line)
        if fence:
            block = self._drain_table()
            if self._in_fence:
                self._in_fence = False
                return block  # the closing fence becomes a blank separator
            self._in_fence = True
            label = f"code:{fence.group(1)}" if fence.group(1) else "code"
            pad = "─" * max(4, 30 - len(label))
            return f"{block}{DIM}─── {label} {pad}{RESET}"

        if self._in_fence:
            return f"{CYAN}{line}{RESET}"

        if self._is_table_row(line):
            # rows are held back so every column can share one width
            self._pending_table.append(line)
            return None

        block = self._drain_table()

        if _RULE.match(line):
            return f"{block}{DIM}{'─' * _RULE_WIDTH}{RESET}"

        heading = _HEADING.match(line)
        if heading:
            style = BOLD + UNDERLINE if len(heading.group(1)) == 1 else BOLD
            body = _inline(heading.group(2).strip(), resume=style)
            return f"{block}{style}{body}{RESET}"

        quote = _QUOTE.match(line)
        if quote:
            style = ITALIC + DIM
            return f"{block}{style}{_inline(quote.group(1), resume=style)}{RESET}"

        listing = _LIST.match(line)
        if listing:
            indent, marker, rest = listing.groups()
            bullet = "•" if marker in "-*+" else marker
            return f"{block}{indent}{YELLOW}{bullet}{RESET} {_inline(rest)}"

        return block + _inline(line)

    @staticmethod
    def _render_table(rows: list[str]) -> str:
        cells_per_row = [
            [cell.strip() for cell in row.strip().strip("|").split("|")]
            for row in rows
        ]
        column_count = max(len(cells) for cells in cells_per_row)
        widths = [0] * column_count
        for cells in cells_per_row:
            for index, cell in enumerate(cells):
                widths[index] = max(widths[index], len(cell))

        # visible width of "  │ " + cells joined by " │ " + " │"
        total = sum(widths) + 3 * column_count + 3

        out: list[str] = []
        for cells in cells_per_row:
            if all(_SEPARATOR.match(cell) for cell in cells):
                out.append(f"{DIM}{'─' * total}{RESET}")
                continue
            padded = [cell.ljust(widths[index]) for index, cell in enumerate(cells)]
            padded += [" " * widths[index]] * (column_count - len(padded))
            body = f" {DIM}│{RESET} ".join(_inline(cell) for cell in padded)
            out.append(f"  {DIM}│{RESET} {body} {DIM}│{RESET}")
        return "".join(f"{row}\n" for row in out)
