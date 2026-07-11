from brain.contracts import SearchResult
from tools import websearch


def test_web_search_keeps_rows_structured_and_uses_cache(monkeypatch):
    calls = []

    def fake_rows(query, top_k, timeout=5):
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


def test_ddgs_client_is_shared_across_searches(monkeypatch):
    """A fresh DDGS per call leaked one HTTP client set (and its sockets) per
    search until file descriptors ran out and SQLite could not open files."""
    instances = []

    class _FakeDDGS:
        def __init__(self, timeout=None):
            instances.append(self)

        def text(self, query, max_results=3):
            return [{"title": f"hit {query}", "href": "https://e.test", "body": "x"}]

    monkeypatch.setattr(websearch, "DDGS", _FakeDDGS)
    websearch._reset_ddgs()

    websearch._search_rows("first query", top_k=3)
    websearch._search_rows("second query", top_k=3)
    websearch._search_rows("third query", top_k=3)

    assert len(instances) == 1
    websearch._reset_ddgs()


def test_ddgs_client_is_rebuilt_after_failure(monkeypatch):
    instances = []

    class _FlakyDDGS:
        def __init__(self, timeout=None):
            self.fails = len(instances) == 0
            instances.append(self)

        def text(self, query, max_results=3):
            if self.fails:
                raise RuntimeError("wedged client")
            return []

    monkeypatch.setattr(websearch, "DDGS", _FlakyDDGS)
    websearch._reset_ddgs()

    try:
        websearch._search_rows("boom", top_k=3)
    except RuntimeError:
        pass
    websearch._search_rows("works", top_k=3)

    assert len(instances) == 2
    websearch._reset_ddgs()
