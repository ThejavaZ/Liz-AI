"""Tests for multiline-aware input (Sprint 1 — terminal experience).

`read_message` decides how lines become one user message: single
lines, backslash continuations, pasted bursts, EOF and Ctrl+C.
"""

import sys

import pytest

from liz.cli import app
from liz.cli.prompt import (
    CONTINUATION_PROMPT,
    read_message,
    stdin_has_pending_lines,
)


def _scripted(*lines):
    """A read() that pops scripted lines and records the prompts used."""
    remaining = list(lines)
    prompts: list[str] = []

    def read(prompt: str) -> str:
        prompts.append(prompt)
        if not remaining:
            raise AssertionError("no scripted line left for " + repr(prompt))
        return remaining.pop(0)

    return read, prompts


class TestSingleLine:
    def test_returns_the_stripped_line(self) -> None:
        read, prompts = _scripted("  hola  ")
        assert read_message(read=read) == "hola"
        assert prompts == ["You: "]

    def test_empty_line_returns_empty(self) -> None:
        read, _ = _scripted("")
        assert read_message(read=read) == ""

    def test_whitespace_only_returns_empty(self) -> None:
        read, _ = _scripted("   ")
        assert read_message(read=read) == ""

    def test_eof_returns_none(self) -> None:
        def read(prompt: str) -> str:
            raise EOFError

        assert read_message(read=read) is None

    def test_keyboard_interrupt_propagates(self) -> None:
        """Ctrl+C at the first prompt keeps the caller's exit behavior."""

        def read(prompt: str) -> str:
            raise KeyboardInterrupt

        with pytest.raises(KeyboardInterrupt):
            read_message(read=read)


class TestBackslashContinuation:
    def test_joins_lines_until_one_without_backslash(self) -> None:
        read, prompts = _scripted("first line\\", "second line")
        assert read_message(read=read) == "first line\nsecond line"
        assert prompts == ["You: ", CONTINUATION_PROMPT]

    def test_multiple_continuations(self) -> None:
        read, _ = _scripted("a\\", "b\\", "c")
        assert read_message(read=read) == "a\nb\nc"

    def test_backslash_is_removed_from_content(self) -> None:
        read, _ = _scripted("foo\\", "bar")
        assert read_message(read=read) == "foo\nbar"

    def test_backslash_only_line_contributes_empty_piece(self) -> None:
        read, _ = _scripted("\\", "body")
        assert read_message(read=read) == "\nbody".strip()

    def test_no_backslash_reads_only_once(self) -> None:
        read, prompts = _scripted("just one")
        assert (
            read_message(read=read, has_pending=lambda: False) == "just one"
        )
        assert len(prompts) == 1

    def test_eof_mid_draft_sends_what_was_read(self) -> None:
        responses = iter(["draft\\"])

        def read(prompt: str) -> str:
            try:
                return next(responses)
            except StopIteration:
                raise EOFError

        assert read_message(read=read) == "draft"

    def test_keyboard_interrupt_mid_draft_cancels(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Ctrl+C mid-draft discards it and stays in the CLI."""

        responses = iter(["draft\\"])

        def read(prompt: str) -> str:
            try:
                return next(responses)
            except StopIteration:
                raise KeyboardInterrupt

        assert read_message(read=read) == ""
        assert capsys.readouterr().out == "\n"


class TestBurstCoalescing:
    def test_pasted_paragraph_arrives_as_one_message(self) -> None:
        """Acceptance: every line of a paste is one user message."""
        read, prompts = _scripted("line one", "line two", "line three")
        pending = iter([True, True, False])

        message = read_message(
            read=read, has_pending=lambda: next(pending)
        )

        assert message == "line one\nline two\nline three"
        assert prompts == ["You: ", CONTINUATION_PROMPT, CONTINUATION_PROMPT]

    def test_not_pending_reads_once(self) -> None:
        read, prompts = _scripted("line one")
        assert (
            read_message(read=read, has_pending=lambda: False)
            == "line one"
        )
        assert len(prompts) == 1

    def test_without_pending_callback_no_coalescing(self) -> None:
        read, prompts = _scripted("only")
        assert read_message(read=read) == "only"
        assert len(prompts) == 1

    def test_continuation_then_pending_drain(self) -> None:
        read, _ = _scripted("a\\", "b", "c")
        pending = iter([True, False])

        message = read_message(
            read=read, has_pending=lambda: next(pending)
        )

        assert message == "a\nb\nc"

    def test_eof_while_draining_keeps_partial(self) -> None:
        responses = iter(["line one"])

        def read(prompt: str) -> str:
            try:
                return next(responses)
            except StopIteration:
                raise EOFError

        assert read_message(read=read, has_pending=lambda: True) == "line one"


class TestBracketMarkers:
    def test_markers_are_stripped(self) -> None:
        read, _ = _scripted("\x1b[200~pasted\x1b[201~")
        assert read_message(read=read) == "pasted"

    def test_markers_around_multiline_paste(self) -> None:
        """inputrc may disable readline's bracketed paste while the
        terminal still wraps the paste; the markers must not leak."""
        read, _ = _scripted("\x1b[200~a\nb\x1b[201~")
        assert read_message(read=read) == "a\nb"


class TestStdinHasPendingLines:
    def test_reports_ready_fd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "liz.cli.prompt.select.select", lambda *a: ([1], [], [])
        )
        assert stdin_has_pending_lines() is True

    def test_reports_empty_fd(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "liz.cli.prompt.select.select", lambda *a: ([], [], [])
        )
        assert stdin_has_pending_lines() is False

    def test_os_error_is_not_pending(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def broken(*a, **kw):
            raise OSError("closed")

        monkeypatch.setattr("liz.cli.prompt.select.select", broken)
        assert stdin_has_pending_lines() is False


class _FakeStdout:
    def __init__(self, tty: bool = True) -> None:
        self.tty = tty
        self.written: list[str] = []

    def isatty(self) -> bool:
        return self.tty

    def write(self, text: str) -> int:
        self.written.append(text)
        return len(text)

    def flush(self) -> None:
        pass


class TestBracketedPasteToggle:
    def test_enables_on_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeStdout(tty=True)
        monkeypatch.setattr(sys, "stdout", fake)
        app._set_bracketed_paste(True)
        assert fake.written == ["\x1b[?200h"]

    def test_disables_on_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeStdout(tty=True)
        monkeypatch.setattr(sys, "stdout", fake)
        app._set_bracketed_paste(False)
        assert fake.written == ["\x1b[?201l"]

    def test_noop_when_not_a_tty(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake = _FakeStdout(tty=False)
        monkeypatch.setattr(sys, "stdout", fake)
        app._set_bracketed_paste(True)
        assert fake.written == []

    def test_noop_without_readline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without readline the markers would leak as literal text."""
        fake = _FakeStdout(tty=True)
        monkeypatch.setattr(sys, "stdout", fake)
        monkeypatch.setattr(app, "HAS_READLINE", False)
        app._set_bracketed_paste(True)
        assert fake.written == []


class TestIsCommandMessage:
    def test_single_line_command(self) -> None:
        assert app.is_command_message("/help") is True

    def test_multiline_paste_is_never_a_command(self) -> None:
        assert app.is_command_message("/help\nextra line") is False

    def test_pasted_paths_belong_to_the_model(self) -> None:
        assert app.is_command_message("/usr/bin\n/usr/lib") is False

    def test_plain_message(self) -> None:
        assert app.is_command_message("hola") is False

    def test_multiline_plain_message(self) -> None:
        assert app.is_command_message("hola\nadios") is False
