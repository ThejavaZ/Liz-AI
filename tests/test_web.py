from unittest.mock import patch

import pytest

from liz.tools.models import ToolStatus
from liz.tools.web import WebSearch, _clean, _clean_url


class TestHelpers:
    def test_clean_strips_tags(self) -> None:
        assert _clean("<b>Hello</b> world") == "Hello world"

    def test_clean_collapses_whitespace(self) -> None:
        assert _clean("  a   b  ") == "a b"

    def test_clean_url_passthrough(self) -> None:
        assert _clean_url("https://example.com") == "https://example.com"

    def test_clean_url_unwraps_redirect(self) -> None:
        wrapped = (
            "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fx"
        )
        assert _clean_url(wrapped) == "https://example.com/x"


class TestWebSearchTool:
    def test_name(self) -> None:
        assert WebSearch().name == "web_search"

    def test_description(self) -> None:
        assert "web" in WebSearch().description.lower()

    def test_parameters_schema(self) -> None:
        schema = WebSearch().parameters_schema
        assert "query" in schema["properties"]
        assert "query" in schema["required"]

    def test_empty_query(self) -> None:
        result = WebSearch().execute(query="")
        assert result.status == ToolStatus.ERROR

    def test_whitespace_query(self) -> None:
        result = WebSearch().execute(query="   ")
        assert result.status == ToolStatus.ERROR

    @patch("liz.tools.web.search_web")
    def test_formats_results(self, mock_search) -> None:
        mock_search.return_value = [
            {
                "title": "Python.org",
                "url": "https://python.org",
                "snippet": "The official Python site.",
            }
        ]

        result = WebSearch().execute(query="python")

        assert result.status == ToolStatus.SUCCESS
        assert "Python.org" in result.data
        assert "https://python.org" in result.data
        assert "official Python site" in result.data

    @patch("liz.tools.web.search_web")
    def test_no_results(self, mock_search) -> None:
        mock_search.return_value = []

        result = WebSearch().execute(query="nothing")

        assert result.status == ToolStatus.SUCCESS
        assert "No results" in result.data

    @patch("liz.tools.web.search_web")
    def test_max_results_clamped(self, mock_search) -> None:
        mock_search.return_value = []

        WebSearch().execute(query="x", max_results=999)
        assert mock_search.call_args[0][1] == 10

        WebSearch().execute(query="x", max_results=0)
        assert mock_search.call_args[0][1] == 1

    @patch("liz.tools.web.search_web")
    def test_search_error_reported(self, mock_search) -> None:
        mock_search.side_effect = RuntimeError("network down")

        result = WebSearch().execute(query="python")

        assert result.status == ToolStatus.ERROR
        assert "failed" in result.error.lower()

    @patch("liz.tools.web.search_web")
    def test_default_max_results(self, mock_search) -> None:
        mock_search.return_value = []
        WebSearch().execute(query="x")
        assert mock_search.call_args[0][1] == 5
