from __future__ import annotations

import re
from typing import Any

from ddgs import DDGS

from brain.config import load_brain_config
from brain.contracts import SearchResult

STOP_TOKENS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "for",
    "to",
    "of",
    "in",
    "on",
    "at",
    "with",
    "from",
    "about",
    "me",
    "my",
    "our",
    "you",
}


def normalize_search_query(query: str) -> str:
    tokens = re.findall(r"[a-z0-9]+", (query or "").lower())
    return " ".join(tokens)


def render_search_results(results: list[SearchResult], max_chars: int = 2200) -> str:
    blocks = []
    for item in results:
        if not item.title:
            continue
        lines = [f"[{item.source_id}] {item.title}"]
        if item.url:
            lines.append(item.url)
        if item.snippet:
            lines.append(item.snippet[:420])
        blocks.append("\n".join(lines))
    text = "\n\n".join(blocks)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"


def _format_results(rows: list[dict[str, Any]], max_chars: int) -> str:
    results = [
        SearchResult.from_row(row, query="", rank=rank)
        for rank, row in enumerate(rows, 1)
    ]
    return render_search_results(results, max_chars=max_chars)


def _search_rows(query: str, top_k: int) -> list[dict[str, str]]:
    rows = []
    with DDGS() as ddgs:
        for item in ddgs.text(query, max_results=max(1, top_k)):
            rows.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("href", ""),
                    "snippet": item.get("body", ""),
                }
            )
    return rows


def _avg_overlap_score(query: str, rows: list[dict[str, Any]]) -> float:
    query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    if not query_tokens or not rows:
        return 0.0
    scores = []
    for row in rows[:2]:
        hay = f"{row.get('title', '')} {row.get('snippet', '')}".lower()
        hit = sum(1 for token in query_tokens if token in hay)
        scores.append(hit / max(1, len(query_tokens)))
    return sum(scores) / len(scores) if scores else 0.0


def _rewrite_query(query: str) -> str:
    tokens = [
        token
        for token in re.findall(r"[a-z0-9]+", query.lower())
        if token not in STOP_TOKENS
    ]
    if not tokens:
        return query.strip()
    return " ".join(tokens[:8])


class WebSearchClient:
    def __init__(self, config: dict[str, Any] | None = None):
        self._config = config
        self._last_error = ""
        self._last_query = ""
        self._last_result_count = 0
        self._checked = False

    def _settings(self) -> dict[str, Any]:
        if self._config is not None:
            return self._config
        return load_brain_config().get("search", {})

    def search(
        self,
        query: str,
        *,
        cache: dict[str, list[SearchResult]] | None = None,
    ) -> list[SearchResult]:
        text = (query or "").strip()
        if not text:
            return []

        cfg = self._settings()
        if not bool(cfg.get("enabled", True)):
            self._last_error = "search is disabled"
            self._checked = True
            return []

        normalized = normalize_search_query(text)
        if cache is not None and normalized in cache:
            cached = list(cache[normalized])
            self._checked = True
            self._last_error = ""
            self._last_query = text
            self._last_result_count = len(cached)
            return cached

        top_k = int(cfg.get("top_k", 3))
        min_overlap = float(cfg.get("min_overlap_score", 0.2))
        self._last_query = text
        self._checked = True
        try:
            rows = _search_rows(text, top_k=top_k)
            original_score = _avg_overlap_score(text, rows)
            if original_score < min_overlap:
                rewritten = _rewrite_query(text)
                if rewritten and rewritten != text:
                    retry_rows = _search_rows(rewritten, top_k=top_k)
                    if _avg_overlap_score(rewritten, retry_rows) > original_score:
                        rows = retry_rows
            self._last_error = ""
        except Exception as exc:
            rows = []
            self._last_error = str(exc)

        results = [
            SearchResult.from_row(row, query=text, rank=rank)
            for rank, row in enumerate(rows, 1)
            if str(row.get("title") or "").strip()
        ]
        self._last_result_count = len(results)
        if cache is not None:
            cache[normalized] = list(results)
        return results

    def health(self) -> dict[str, Any]:
        enabled = bool(self._settings().get("enabled", True))
        available = None
        state = "unknown"
        if not enabled:
            available = False
            state = "disabled"
        elif self._checked:
            available = not self._last_error
            state = "healthy" if available else "unavailable"
        return {
            "available": available,
            "state": state,
            "enabled": enabled,
            "last_error": self._last_error,
            "last_query": self._last_query,
            "last_result_count": self._last_result_count,
        }


DEFAULT_SEARCH_CLIENT = WebSearchClient()


def search_web(
    query: str,
    cache: dict[str, list[SearchResult]] | None = None,
) -> list[SearchResult]:
    return DEFAULT_SEARCH_CLIENT.search(query, cache=cache)


def get_search_health() -> dict[str, Any]:
    return DEFAULT_SEARCH_CLIENT.health()


def tool_websearch(
    query: str,
    cache: dict[str, list[SearchResult]] | None = None,
) -> str:
    cfg = load_brain_config().get("search", {})
    max_chars = int(cfg.get("max_chars", 2200))
    return render_search_results(search_web(query, cache=cache), max_chars=max_chars)
