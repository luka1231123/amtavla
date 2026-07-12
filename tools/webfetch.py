"""WEB_FETCH (T0): fetch one URL and return readable text + a citation.

Read-only and bounded — timeout, size cap, http(s) only, scripts/styles stripped.
The fetcher is injectable so the turn loop can be exercised offline without the
network. Fetched content becomes a `web:<hash>` source, exactly like SEARCH, so
answers cite it the same way.
"""

from __future__ import annotations

import re
import urllib.request
from html import unescape
from typing import Any, Callable

from brain.contracts import SourceRef, stable_source_id

_DEFAULT_TIMEOUT = 8
_MAX_FETCH_BYTES = 1_500_000
_MAX_TEXT_CHARS = 6000
_USER_AGENT = "amtavla/1.0 (+local-first assistant)"

_SCRIPT_STYLE_RE = re.compile(r"<(script|style|noscript)[^>]*>.*?</\1>", re.I | re.S)
_TAG_RE = re.compile(r"<[^>]+>")
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_MULTINL_RE = re.compile(r"\n{3,}")


def _default_fetcher(url: str, timeout: int) -> tuple[str, bytes]:
    """Return (content_type, body_bytes). Isolated so tests can replace it."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        body = response.read(_MAX_FETCH_BYTES + 1)
    return content_type, body


def html_to_text(html: str) -> str:
    without_scripts = _SCRIPT_STYLE_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", without_scripts)
    text = unescape(text)
    lines = [_WS_RE.sub(" ", line).strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)
    return _MULTINL_RE.sub("\n\n", text).strip()


class WebFetchClient:
    def __init__(
        self,
        *,
        fetcher: Callable[[str, int], tuple[str, bytes]] | None = None,
        timeout: int = _DEFAULT_TIMEOUT,
        max_chars: int = _MAX_TEXT_CHARS,
    ) -> None:
        self._fetch = fetcher or _default_fetcher
        self.timeout = timeout
        self.max_chars = max_chars

    def fetch(self, url: str) -> dict[str, Any]:
        target = (url or "").strip()
        if not re.match(r"^https?://[^\s]+\.[^\s]+", target, re.IGNORECASE):
            return {"operation": "web_fetch", "error": f"Not a valid http(s) URL: {url}"}
        try:
            content_type, body = self._fetch(target, self.timeout)
        except Exception as exc:
            return {"operation": "web_fetch", "url": target, "error": f"Fetch failed: {exc}"}

        truncated_bytes = len(body) > _MAX_FETCH_BYTES
        raw = body[:_MAX_FETCH_BYTES].decode("utf-8", errors="replace")
        title = ""
        if "html" in content_type.lower() or "<html" in raw[:2000].lower():
            match = _TITLE_RE.search(raw)
            if match:
                title = _WS_RE.sub(" ", unescape(_TAG_RE.sub("", match.group(1)))).strip()
            text = html_to_text(raw)
        else:
            text = raw.strip()
        truncated = truncated_bytes or len(text) > self.max_chars
        text = text[: self.max_chars]
        if not text:
            return {"operation": "web_fetch", "url": target, "error": "No readable text at that URL"}
        return {
            "operation": "web_fetch",
            "url": target,
            "title": title or target,
            "content": text,
            "truncated": truncated,
            "source_id": stable_source_id("web", target),
        }

    def source_for(self, result: dict[str, Any]) -> SourceRef:
        return SourceRef(
            source_id=result["source_id"],
            kind="web",
            title=result.get("title") or result["url"],
            excerpt=str(result.get("content", ""))[:200],
            url=result["url"],
            metadata={"tier": "T0", "fetched": True},
        )
