"""Web search via DuckDuckGo.

DuckDuckGo's HTML endpoint needs no API key, so this tool works out of
the box. The dependency is optional: if the search library is missing
the tool reports a clear message instead of crashing.
"""

import json
import urllib.parse
import urllib.request
import warnings
from typing import Any

from liz.tools.base import Tool
from liz.tools.models import ToolResult

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
TIMEOUT_SECONDS = 10
MAX_RESULTS = 5


def _load_ddgs():
    """Import DDGS from either package name.

    `duckduckgo_search` was renamed to `ddgs`; both spellings are
    supported so whichever one is installed works.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        from duckduckgo_search import DDGS
    return DDGS


def _search_with_library(query: str, max_results: int) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    original_warn = warnings.warn

    def _quiet(message, *args, **kwargs):
        # The legacy package forces its rename warning through with
        # simplefilter("always"), which would bypass any filter and land
        # in the middle of the streamed reply. Drop just that message.
        if "renamed to `ddgs`" in str(message):
            return None
        return original_warn(message, *args, **kwargs)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*renamed.*")
        warnings.warn = _quiet
        try:
            ddgs_class = _load_ddgs()
            with ddgs_class() as ddgs:
                for item in ddgs.text(query, max_results=max_results):
                    results.append(
                        {
                            "title": item.get("title", ""),
                            "url": item.get("href", ""),
                            "snippet": item.get("body", ""),
                        }
                    )
        finally:
            warnings.warn = original_warn
    return results


def _fetch_html(url: str, data: bytes | None = None) -> str:
    headers = {"User-Agent": USER_AGENT}
    if data is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
        return response.read().decode("utf-8", errors="replace")


def _search_fallback(query: str, max_results: int) -> list[dict[str, str]]:
    """Minimal fallback using the DuckDuckGo HTML endpoint.

    The endpoint expects the query as a POST body; sending it as a GET
    returns the landing page and zero results.
    """
    url = "https://html.duckduckgo.com/html/"
    body = urllib.parse.urlencode({"q": query}).encode()
    try:
        html = _fetch_html(url, data=body)
    except Exception:
        # some deployments still answer plain GET
        html = _fetch_html(url + "?" + urllib.parse.urlencode({"q": query}))

    results: list[dict[str, str]] = []
    for chunk in html.split('class="result__a"')[1:]:
        if len(results) >= max_results:
            break
        href = _extract_attr(chunk, "href")
        if "duckduckgo.com/y.js" in href:
            continue  # sponsored result
        results.append(
            {
                "title": _clean(_extract_between(chunk, ">", "</a>")),
                "url": _clean_url(href),
                "snippet": _clean(
                    _extract_between(chunk, 'class="result__snippet"', "</a>")
                ),
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
    """Search the web without an API key.

    The HTML endpoint is the reliable path, so it goes first; the legacy
    library (when installed) is kept as a backup because it answers 0
    results or homepage junk for some queries.
    """
    try:
        results = _search_fallback(query, max_results)
    except Exception:
        results = []
    if results:
        return results

    try:
        return _search_with_library(query, max_results)
    except Exception:
        return []


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
