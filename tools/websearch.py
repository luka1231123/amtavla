import json
import re
import time
import urllib.parse
import urllib.request
from urllib.error import HTTPError

from ddgs import DDGS

MAX_RESULTS = 3
MAX_BODY_WORDS = 200
RATE_LIMIT_COOLDOWN_SECONDS = 60
_RATE_LIMIT_UNTIL = {"wikipedia": 0.0, "wikidata": 0.0}


STOPWORDS = {
    "what",
    "who",
    "when",
    "where",
    "why",
    "how",
    "is",
    "are",
    "the",
    "a",
    "an",
    "of",
    "to",
    "for",
    "in",
    "on",
    "about",
    "with",
    "and",
    "tell",
    "me",
    "please",
    "explain",
    "show",
}


def _fetch_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "amtavla/1.0"})
    with urllib.request.urlopen(req, timeout=12) as response:
        return json.loads(response.read().decode("utf-8", errors="replace"))


def _simplify_query(query: str) -> str:
    tokens = re.findall(r"[a-zA-Z0-9']+", query or "")
    filtered = [t for t in tokens if t.lower() not in STOPWORDS]
    if filtered and filtered[0].lower() in {"listen", "ok", "okay", "hey", "yo"}:
        filtered = filtered[1:]
    if not filtered:
        return query.strip()
    return " ".join(filtered[:6])


def _entity_query(query: str) -> str:
    tokens = re.findall(r"[a-zA-Z0-9']+", query or "")
    if not tokens:
        return ""
    caps = [t for t in tokens if t and t[0].isupper() and t.lower() not in STOPWORDS]
    if len(caps) >= 2:
        return " ".join(caps[:2])
    plain = [t for t in tokens if t.lower() not in STOPWORDS]
    return " ".join((plain or tokens)[:2])


def _wiki_result(query: str) -> list[dict]:
    q = urllib.parse.quote(query)
    search_url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={q}&limit=1&namespace=0&format=json"
    data = _fetch_json(search_url)
    titles = data[1] if isinstance(data, list) and len(data) > 1 else []
    if not titles:
        return []
    title = titles[0]
    summary_url = (
        "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title)
    )
    summary = _fetch_json(summary_url)
    extract = summary.get("extract", "")
    page = summary.get("content_urls", {}).get("desktop", {}).get("page", "")
    return [{"title": title, "href": page, "body": extract}]


def _wikidata_result(query: str) -> list[dict]:
    q = urllib.parse.quote(query)
    url = (
        "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json"
        f"&language=en&type=item&limit=1&search={q}"
    )
    data = _fetch_json(url)
    results = data.get("search", [])
    if not results:
        return []
    item = results[0]
    qid = item.get("id", "")
    title = item.get("label", qid)
    desc = item.get("description", "")
    href = f"https://www.wikidata.org/wiki/{qid}" if qid else ""
    return [{"title": title, "href": href, "body": desc}]


def _provider_blocked(provider: str) -> bool:
    return time.time() < _RATE_LIMIT_UNTIL.get(provider, 0.0)


def _mark_rate_limited(provider: str):
    _RATE_LIMIT_UNTIL[provider] = time.time() + RATE_LIMIT_COOLDOWN_SECONDS


def deep_crawl_websearch(query: str, max_results: int = 8) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        return f"Deep crawl failed: {e}"

    if not results:
        return "No deep crawl results found."

    parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        url = r.get("href", "")
        body = r.get("body", "")
        words = body.split()
        if len(words) > MAX_BODY_WORDS:
            body = " ".join(words[:MAX_BODY_WORDS]) + "..."
        parts.append(f"[{i}] {title}\n    {url}\n    {body}")
    return "\n\n".join(parts)


def tool_websearch(query: str) -> str:
    base_query = (query or "").strip()
    fallback_query = _simplify_query(base_query)
    entity_query = _entity_query(base_query)
    starts_question = bool(
        re.match(r"^(what|who|when|where|why|how)\b", base_query.lower())
    )

    candidates = []
    if starts_question:
        candidates.extend([entity_query, fallback_query, base_query])
    else:
        candidates.extend([base_query, fallback_query, entity_query])
    ordered_queries = []
    for q in candidates:
        q = (q or "").strip()
        if q and q not in ordered_queries:
            ordered_queries.append(q)

    results = []
    if _provider_blocked("wikipedia"):
        results.append(
            {
                "title": "Wikipedia",
                "href": "",
                "body": "Wikipedia lookup skipped due to recent rate limit cooldown.",
            }
        )
    else:
        try:
            for q in ordered_queries:
                results.extend(_wiki_result(q))
                if results and results[-1].get("title") != "Wikipedia":
                    break
        except HTTPError as e:
            if e.code == 429:
                _mark_rate_limited("wikipedia")
            results.append(
                {
                    "title": "Wikipedia",
                    "href": "",
                    "body": f"Wikipedia lookup failed: HTTP Error {e.code}",
                }
            )
        except Exception as e:
            results.append(
                {
                    "title": "Wikipedia",
                    "href": "",
                    "body": f"Wikipedia lookup failed: {e}",
                }
            )

    if _provider_blocked("wikidata"):
        results.append(
            {
                "title": "Wikidata",
                "href": "",
                "body": "Wikidata lookup skipped due to recent rate limit cooldown.",
            }
        )
    else:
        try:
            for q in ordered_queries:
                if (
                    len(
                        [
                            x
                            for x in results
                            if x.get("title") not in {"Wikipedia", "Wikidata"}
                        ]
                    )
                    >= 2
                ):
                    break
                results.extend(_wikidata_result(q))
                if results and results[-1].get("title") != "Wikidata":
                    break
        except HTTPError as e:
            if e.code == 429:
                _mark_rate_limited("wikidata")
            results.append(
                {
                    "title": "Wikidata",
                    "href": "",
                    "body": f"Wikidata lookup failed: HTTP Error {e.code}",
                }
            )
        except Exception as e:
            results.append(
                {
                    "title": "Wikidata",
                    "href": "",
                    "body": f"Wikidata lookup failed: {e}",
                }
            )

    if not results:
        return "No curated web results found."

    parts = []
    for i, r in enumerate(results[:MAX_RESULTS], 1):
        title = r.get("title", "")
        url = r.get("href", "")
        body = r.get("body", "")
        words = body.split()
        if len(words) > MAX_BODY_WORDS:
            body = " ".join(words[:MAX_BODY_WORDS]) + "..."
        parts.append(f"[{i}] {title}\n    {url}\n    {body}")
    return "\n\n".join(parts)
