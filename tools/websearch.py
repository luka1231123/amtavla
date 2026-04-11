import re

from ddgs import DDGS

from brain.config import load_brain_config

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


def _format_results(rows: list[dict], max_chars: int) -> str:
    blocks = []
    for idx, row in enumerate(rows, 1):
        title = (row.get("title") or "").strip()
        if not title:
            continue
        url = (row.get("url") or "").strip()
        snippet = (row.get("snippet") or "").strip()
        lines = [f"[{idx}] {title}"]
        if url:
            lines.append(url)
        if snippet:
            lines.append(snippet[:420])
        blocks.append("\n".join(lines))
    text = "\n\n".join(blocks)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"


def _search_rows(query: str, top_k: int) -> list[dict]:
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


def _avg_overlap_score(query: str, rows: list[dict]) -> float:
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
        t for t in re.findall(r"[a-z0-9]+", query.lower()) if t not in STOP_TOKENS
    ]
    if not tokens:
        return query.strip()
    return " ".join(tokens[:8])


def tool_websearch(query: str, cache: dict | None = None) -> str:
    text = (query or "").strip()
    if not text:
        return ""

    cfg = load_brain_config().get("search", {})
    if not bool(cfg.get("enabled", True)):
        return ""
    top_k = int(cfg.get("top_k", 3))
    max_chars = int(cfg.get("max_chars", 2200))
    min_overlap = float(cfg.get("min_overlap_score", 0.2))

    normalized = normalize_search_query(text)
    if cache is not None and normalized in cache:
        return cache[normalized]

    try:
        rows = _search_rows(text, top_k=top_k)
        if _avg_overlap_score(text, rows) < min_overlap:
            rewritten = _rewrite_query(text)
            if rewritten and rewritten != text:
                retry_rows = _search_rows(rewritten, top_k=top_k)
                if _avg_overlap_score(rewritten, retry_rows) > _avg_overlap_score(
                    text, rows
                ):
                    rows = retry_rows
    except Exception:
        rows = []

    result = _format_results(rows, max_chars=max_chars)
    if cache is not None:
        cache[normalized] = result
    return result
