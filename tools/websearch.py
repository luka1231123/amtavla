from ddgs import DDGS

MAX_RESULTS = 3
MAX_BODY_WORDS = 200


def tool_websearch(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=MAX_RESULTS))
    except Exception as e:
        return f"Websearch failed: {e}"

    if not results:
        return "No web results found."

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
