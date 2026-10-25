import sys
import types
import warnings
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest

from liz.tools.models import ToolStatus
from liz.tools.web import (
    WebSearch,
    _clean,
    _clean_url,
    _load_ddgs,
    _search_fallback,
    _search_with_library,
    search_web,
)


def _fake_module(name: str, ddgs_cls) -> types.ModuleType:
    module = types.ModuleType(name)
    module.DDGS = ddgs_cls
    return module


class _FakeDDGS:
    """Stands in for the real client and warns like the legacy package."""

    def __init__(self) -> None:
        # the real package does exactly this before warning, which
        # would bypass any filter set by the caller
        warnings.simplefilter("always")
        warnings.warn(
            "This package (`duckduckgo_search`) has been renamed to `ddgs`!",
            RuntimeWarning,
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def text(self, query, max_results=5):
        return [
            {"title": "T", "href": "https://x.test", "body": "B"}
        ][:max_results]


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


class TestDDGSLibrary:
    """`duckduckgo_search` was renamed to `ddgs`; both must work quietly."""

    def test_prefers_ddgs_package(self, monkeypatch) -> None:
        monkeypatch.setitem(
            sys.modules, "ddgs", _fake_module("ddgs", _FakeDDGS)
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = _search_with_library("python", 3)

        assert results[0]["url"] == "https://x.test"
        assert caught == []

    def test_falls_back_to_legacy_package(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "ddgs", None)
        monkeypatch.setitem(
            sys.modules,
            "duckduckgo_search",
            _fake_module("duckduckgo_search", _FakeDDGS),
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = _search_with_library("python", 3)

        assert results[0]["title"] == "T"
        assert caught == []

    def test_rename_warning_is_suppressed(self, monkeypatch) -> None:
        monkeypatch.setitem(
            sys.modules, "ddgs", _fake_module("ddgs", _FakeDDGS)
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            _search_with_library("x", 1)

        assert not [
            w for w in caught if "renamed" in str(w.message)
        ]

    def test_missing_both_packages_raises(self, monkeypatch) -> None:
        monkeypatch.setitem(sys.modules, "ddgs", None)
        monkeypatch.setitem(sys.modules, "duckduckgo_search", None)

        with pytest.raises(ImportError):
            _load_ddgs()


class TestSearchSelection:
    @patch("liz.tools.web._search_with_library")
    @patch("liz.tools.web._search_fallback")
    def test_html_result_wins(self, mock_html, mock_lib) -> None:
        hits = [{"title": "T", "url": "u", "snippet": "s"}]
        mock_html.return_value = hits

        assert search_web("q") == hits
        mock_lib.assert_not_called()

    @patch("liz.tools.web._search_with_library")
    @patch("liz.tools.web._search_fallback")
    def test_empty_html_uses_library(self, mock_html, mock_lib) -> None:
        hits = [{"title": "T", "url": "u", "snippet": "s"}]
        mock_html.return_value = []
        mock_lib.return_value = hits

        assert search_web("q") == hits
        mock_lib.assert_called_once()

    @patch("liz.tools.web._search_with_library")
    @patch("liz.tools.web._search_fallback")
    def test_html_error_uses_library(self, mock_html, mock_lib) -> None:
        mock_html.side_effect = URLError("down")
        mock_lib.return_value = []

        assert search_web("q") == []

    @patch("liz.tools.web._search_with_library")
    @patch("liz.tools.web._search_fallback")
    def test_both_paths_failing_returns_empty(
        self, mock_html, mock_lib
    ) -> None:
        mock_html.side_effect = URLError("down")
        mock_lib.side_effect = ImportError("no package")

        assert search_web("q") == []


class TestHtmlFallback:
    @patch("liz.tools.web.urllib.request.urlopen")
    def test_posts_the_query(self, mock_urlopen) -> None:
        html = b'<a class="result__a" href="https://x.test">Title</a>'
        response = MagicMock()
        response.__enter__.return_value.read.return_value = html
        mock_urlopen.return_value = response

        results = _search_fallback("python rocks", 5)

        request = mock_urlopen.call_args[0][0]
        assert request.data == b"q=python+rocks"
        assert request.get_header("Content-type") == (
            "application/x-www-form-urlencoded"
        )
        assert results[0]["url"] == "https://x.test"
        assert results[0]["title"] == "Title"

    @patch("liz.tools.web.urllib.request.urlopen")
    def test_post_failure_retries_with_get(self, mock_urlopen) -> None:
        html = b'<a class="result__a" href="https://y.test">Other</a>'
        response = MagicMock()
        response.__enter__.return_value.read.return_value = html
        mock_urlopen.side_effect = [URLError("down"), response]

        results = _search_fallback("q", 5)

        assert results[0]["url"] == "https://y.test"
        assert mock_urlopen.call_args_list[1][0][0].data is None

    @patch("liz.tools.web.urllib.request.urlopen")
    def test_skips_sponsored_results(self, mock_urlopen) -> None:
        html = (
            b'<a class="result__a" href="https://duckduckgo.com/y.js?ad=x">Ad</a>'
            b'<a class="result__a" href="https://real.test/">Organic</a>'
        )
        response = MagicMock()
        response.__enter__.return_value.read.return_value = html
        mock_urlopen.return_value = response

        results = _search_fallback("q", 5)

        assert [item["url"] for item in results] == ["https://real.test/"]

    @patch("liz.tools.web.urllib.request.urlopen")
    def test_max_results_respected(self, mock_urlopen) -> None:
        html = "".join(
            f'<a class="result__a" href="https://r{i}.test/">T{i}</a>'
            for i in range(10)
        ).encode()
        response = MagicMock()
        response.__enter__.return_value.read.return_value = html
        mock_urlopen.return_value = response

        assert len(_search_fallback("q", 3)) == 3
