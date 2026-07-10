from brain.contracts import SearchResult
from tools import websearch


def test_web_search_keeps_rows_structured_and_uses_cache(monkeypatch):
    calls = []

    def fake_rows(query, top_k):
        calls.append((query, top_k))
        return [
            {
                "title": "Typed turn loop",
                "url": "https://example.test/turn-loop",
                "snippet": "A typed turn loop keeps execution observable.",
            }
        ]

    monkeypatch.setattr(websearch, "_search_rows", fake_rows)
    client = websearch.WebSearchClient(
        {"enabled": True, "top_k": 3, "min_overlap_score": 0.0}
    )
    cache = {}

    assert client.health()["available"] is None
    assert client.health()["state"] == "unknown"

    first = client.search("typed turn loop", cache=cache)
    second = client.search("typed turn loop", cache=cache)

    assert len(calls) == 1
    assert first == second
    assert isinstance(first[0], SearchResult)
    assert first[0].source_id.startswith("web:")
    assert "[web:" in websearch.render_search_results(first)
    assert client.health()["last_result_count"] == 1
