"""Web search via DuckDuckGo.

DuckDuckGo's HTML endpoint needs no API key, so this tool works out of
the box. The dependency is optional: if the search library is missing
the tool reports a clear message instead of crashing.
"""

import json
import urllib.parse
import urllib.request
from typing import Any

from liz.tools.base import Tool
from liz.tools.models import ToolResult

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
TIMEOUT_SECONDS = 10
MAX_RESULTS = 5


def _search_with_library(query: str, max_results: int) -> list[dict[str, str]]:
    from duckduckgo_search import DDGS

    results: list[dict[str, str]] = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", ""),
                }
            )
    return results


def _search_fallback(query: str, max_results: int) -> list[dict[str, str]]:
    """Minimal fallback using the DuckDuckGo lite HTML endpoint."""
    url = "https://html.duckduckgo.com/html/" + urllib.parse.urlencode(
        {"q": query}
    )
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        html = response.read().decode("utf-8", errors="replace")

    results: list[dict[str, str]] = []
    for chunk in html.split('class="result__a"')[1: max_results + 1]:
        title = _extract_between(chunk, ">", "</a>")
        href = _extract_attr(chunk, "href")
        snippet = _extract_between(chunk, 'class="result__snippet"', "</a>")
        results.append(
            {
                "title": _clean(title),
                "url": _clean_url(href),
                "snippet": _clean(snippet),
            }
        )
    return results


def _extract_between(text: str, start: str, end: str) -> str:
    try:
        begin = text.index(start) + len(start)
        finish = text.index(end, begin)
        return text[begin:finish]
    except ValueError:
        return ""


def _extract_attr(text: str, attr: str) -> str:
    marker = f'{attr}="'
    try:
        begin = text.index(marker) + len(marker)
        finish = text.index('"', begin)
        return text[begin:finish]
    except ValueError:
        return ""


def _clean(text: str) -> str:
    text = text.replace("<b>", "").replace("</b>", "")
    text = text.replace("<strong>", "").replace("</strong>", "")
    text = " ".join(text.split())
    return text.strip()


def _clean_url(url: str) -> str:
    if "//duckduckgo.com/l/" in url and "uddg=" in url:
        try:
            query = urllib.parse.urlparse(url).query
            target = urllib.parse.parse_qs(query).get("uddg", [""])[0]
            if target:
                return urllib.parse.unquote(target)
        except Exception:
            pass
    return url


def search_web(query: str, max_results: int = MAX_RESULTS) -> list[dict[str, str]]:
    """Search the web, preferring the library and falling back to HTML."""
    try:
        return _search_with_library(query, max_results)
    except ImportError:
        return _search_fallback(query, max_results)
    except Exception:
        return _search_fallback(query, max_results)


class WebSearch(Tool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web for current information"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (default 5)",
                },
            },
            "required": ["query"],
        }

    def execute(
        self, query: str, max_results: int = MAX_RESULTS
    ) -> ToolResult:
        if not query or not query.strip():
            return ToolResult.error("Query is empty")

        max_results = max(1, min(int(max_results), 10))

        try:
            results = search_web(query.strip(), max_results)
        except Exception as e:
            return ToolResult.error(f"Web search failed: {e}")

        if not results:
            return ToolResult.success(f"No results found for: {query}")

        lines = [f"Results for: {query}", ""]
        for index, item in enumerate(results, start=1):
            lines.append(f"{index}. {item['title']}")
            if item["url"]:
                lines.append(f"   {item['url']}")
            if item["snippet"]:
                lines.append(f"   {item['snippet']}")
            lines.append("")

        return ToolResult.success("\n".join(lines).strip())
