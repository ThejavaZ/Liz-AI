from liz.cli.markdown import (
    BOLD,
    CYAN,
    DIM,
    ITALIC,
    RESET,
    UNDERLINE,
    YELLOW,
    MarkdownRenderer,
    supports_color,
)


def _render(chunks: list[str], enabled: bool = True) -> str:
    renderer = MarkdownRenderer(enabled=enabled)
    out = "".join(renderer.feed(chunk) for chunk in chunks)
    return out + renderer.flush()


class TestStreaming:
    def test_buffers_partial_line(self) -> None:
        renderer = MarkdownRenderer(enabled=True)
        assert renderer.feed("hola") == ""

    def test_flush_emits_pending_text(self) -> None:
        renderer = MarkdownRenderer(enabled=True)
        renderer.feed("hola")
        assert renderer.flush() == "hola"

    def test_flush_is_empty_when_nothing_pending(self) -> None:
        renderer = MarkdownRenderer(enabled=True)
        renderer.feed("hola\n")
        assert renderer.flush() == ""

    def test_emits_completed_lines(self) -> None:
        out = _render(["uno", "\ndos\n", "tres"])
        assert out == "uno\ndos\ntres"

    def test_streaming_matches_single_shot(self) -> None:
        text = "# Titulo\n\nHola **mundo**, te dejo `codigo`.\n"
        chunked = _render([text[i:i + 3] for i in range(0, len(text), 3)])
        whole = _render([text])
        assert chunked == whole

    def test_single_chunk_without_newline_is_flushed(self) -> None:
        assert _render(["solo"]) == "solo"


class TestInline:
    def test_bold(self) -> None:
        out = _render(["**fuerte**"])
        assert out == f"{BOLD}fuerte{RESET}"

    def test_italic(self) -> None:
        out = _render(["*suave*"])
        assert out == f"{ITALIC}suave{RESET}"

    def test_inline_code(self) -> None:
        out = _render(["usa `ls` aqui"])
        assert out == f"usa {CYAN}ls{RESET} aqui"

    def test_strike(self) -> None:
        out = _render(["~~borrado~~"])
        assert out == f"{DIM}borrado{RESET}"

    def test_link(self) -> None:
        out = _render(["[docs](https://x.com)"])
        assert out == f"docs {DIM}(https://x.com){RESET}"

    def test_underscore_word_kept(self) -> None:
        assert _render(["nombre_variable"]) == "nombre_variable"

    def test_plain_text_untouched(self) -> None:
        assert _render(["texto normal"]) == "texto normal"


class TestBlockElements:
    def test_heading_level_one_underlined(self) -> None:
        out = _render(["# Titulo"])
        style = BOLD + UNDERLINE
        assert out == f"{style}Titulo{RESET}"

    def test_heading_hashes_removed(self) -> None:
        out = _render(["## Segundo"])
        assert "#" not in out
        assert f"{BOLD}Segundo{RESET}" == out

    def test_heading_keeps_style_past_inline(self) -> None:
        out = _render(["# Hola **mundo**"])
        style = BOLD + UNDERLINE
        assert out.startswith(f"{style}Hola {BOLD}mundo")
        assert out.endswith(RESET)

    def test_bullet_list(self) -> None:
        out = _render(["- uno"])
        assert out == f"{YELLOW}•{RESET} uno"

    def test_numbered_list_keeps_number(self) -> None:
        out = _render(["1. primero"])
        assert out.startswith(f"{YELLOW}1.{RESET} primero")

    def test_quote_is_dim(self) -> None:
        out = _render(["> citado"])
        style = ITALIC + DIM
        assert out == f"{style}citado{RESET}"

    def test_rule_becomes_line(self) -> None:
        out = _render(["---"])
        assert out == f"{DIM}{'─' * 40}{RESET}"

    def test_blank_line_preserved(self) -> None:
        assert _render(["a\n\nb"]) == "a\n\nb"


class TestCodeFence:
    def test_fence_marker_and_content(self) -> None:
        out = _render(["```python\nprint(1)\n```\n"])
        lines = out.splitlines()
        assert f"{DIM}" in lines[0] and "code:python" in lines[0]
        assert lines[1] == f"{CYAN}print(1){RESET}"
        assert lines[2] == ""  # closing fence becomes a blank separator

    def test_content_is_not_markdown_processed(self) -> None:
        out = _render(["```\n**no bold**\n```\n"])
        assert f"{CYAN}**no bold**{RESET}" in out

    def test_plain_fence_label(self) -> None:
        out = _render(["```\nx\n"])
        assert "code " in out.splitlines()[0]


class TestTables:
    def test_separator_row_matches_width(self) -> None:
        out = _render(["| --- | --- |"])
        assert out == f"{DIM}{'─' * 15}{RESET}"

    def test_data_row_aligned(self) -> None:
        out = _render(["| a | bb |"])
        assert "│" in out
        assert f"{DIM}│{RESET}" in out
        assert "a " in out and "bb" in out

    def test_multiline_alignment_consistent(self) -> None:
        out = _render(["| a | bb |\n| --- | --- |\n| cc | d |\n"])
        rows = [row for row in out.splitlines() if row]
        widths = {len(row) for row in rows if "│" in row}
        assert len(widths) == 1

    def test_rows_share_column_widths(self) -> None:
        out = _render(["| a |\n| --- |\n| mucho texto |\n"])
        data = [row for row in out.splitlines() if "│" in row]
        assert len({len(row) for row in data}) == 1

    def test_table_flushed_before_next_block(self) -> None:
        out = _render(["| a |\n| --- |\ntexto\n"])
        lines = [line for line in out.splitlines() if line]
        assert lines[-1] == "texto"
        assert any("a" in line for line in lines[:-1])

    def test_table_at_end_of_stream_flushes(self) -> None:
        out = _render(["| a |"])
        assert "│" in out


class TestDisabled:
    def test_passthrough(self) -> None:
        text = "# hola **mundo**\n"
        assert _render([text], enabled=False) == text

    def test_flush_disabled_is_empty(self) -> None:
        renderer = MarkdownRenderer(enabled=False)
        renderer.feed("hola")
        assert renderer.flush() == ""

    def test_no_color_env(self, monkeypatch) -> None:
        monkeypatch.setenv("NO_COLOR", "1")
        assert supports_color(stream=_Tty()) is False

    def test_dumb_terminal(self, monkeypatch) -> None:
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "dumb")
        assert supports_color(stream=_Tty()) is False

    def test_tty_enables(self, monkeypatch) -> None:
        monkeypatch.delenv("NO_COLOR", raising=False)
        monkeypatch.setenv("TERM", "xterm-256color")
        assert supports_color(stream=_Tty()) is True

    def test_non_tty_disables(self) -> None:
        assert supports_color(stream=object()) is False


class _Tty:
    def isatty(self) -> bool:
        return True
