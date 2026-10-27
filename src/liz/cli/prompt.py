"""Multiline-aware user input for the CLI prompt.

A bare ``input("You: ")`` splits a pasted paragraph into several
messages: every newline returns a line, and the leftovers are consumed
by the next prompt as separate inputs (sometimes even as commands).

Three mechanisms make a paragraph arrive as a single message:

1. ``import readline`` gives line editing and bracketed paste (readline
   >= 7). Once the terminal is asked to wrap pastes in
   ``ESC[200~ ... ESC[201~``, readline delivers the whole paste in one
   ``input()`` call — still editable, sent on an explicit Enter.
2. Burst coalescing: on a TTY without bracketed paste the remaining
   lines of a paste are already pending on the stdin fd, so they are
   drained into the same message (``select`` with a zero timeout, so
   normal typing never waits).
3. Explicit continuation: a line ending in a backslash keeps reading
   at a ``... `` prompt until a line without one.

Ctrl+C during a continuation cancels the draft and returns to the
prompt; Ctrl+C at the first prompt propagates so the caller keeps its
exit behavior; Ctrl+D returns ``None`` (EOF).
"""

from __future__ import annotations

import select
import sys
from collections.abc import Callable

CONTINUATION_PROMPT = "... "

# Bracketed-paste markers can leak through as literal text when the
# terminal wraps the paste but readline's bracketed paste is off (e.g.
# `set enable-bracketed-paste off` in inputrc).
_BRACKET_START = "\x1b[200~"
_BRACKET_END = "\x1b[201~"


def stdin_has_pending_lines() -> bool:
    """True when more input is already waiting on the stdin fd.

    Used to coalesce the lines of a paste on terminals that do not use
    bracketed paste. The zero timeout keeps typing latency at zero:
    the bytes of a burst are either already there or they are not.
    """
    try:
        ready, _, _ = select.select([sys.stdin], [], [], 0)
    except (OSError, ValueError):
        return False
    return bool(ready)


def _clean(line: str) -> str:
    return line.replace(_BRACKET_START, "").replace(_BRACKET_END, "")


def read_message(
    read: Callable[[str], str] = input,
    has_pending: Callable[[], bool] | None = None,
    prompt: str = "You: ",
) -> str | None:
    """Read one user message, joining continuation and pasted lines.

    Returns ``None`` when the first prompt hits EOF (Ctrl+D), ``""``
    when a draft is cancelled with Ctrl+C, and otherwise the stripped
    message text (EOF mid-draft keeps what was already read).
    """
    try:
        first = read(prompt)
    except EOFError:
        return None
    # KeyboardInterrupt propagates: Ctrl+C at the first prompt is the
    # caller's to handle (exit), not a cancelled draft.

    lines = [_clean(first)]
    while True:
        if lines[-1].endswith("\\"):
            lines[-1] = lines[-1][:-1]
        elif has_pending is None or not has_pending():
            break

        try:
            lines.append(_clean(read(CONTINUATION_PROMPT)))
        except EOFError:
            break
        except KeyboardInterrupt:
            # Ctrl+C mid-draft: discard it instead of leaving the CLI.
            print()
            return ""

    return "\n".join(lines).strip()
