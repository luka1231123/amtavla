from brain.action_runner import ActionRunner, calculate
from brain.contracts import Action, ActionType, SearchResult


class _FakeSearchClient:
    def search(self, query, *, cache=None):
        result = SearchResult.from_row(
            {
                "title": "Amtavla architecture",
                "url": "https://example.test/amtavla",
                "snippet": "A typed cognitive loop.",
            },
            query=query,
            rank=1,
        )
        if cache is not None:
            cache[query] = [result]
        return [result]

    def health(self):
        return {"available": True, "last_error": ""}


class _FakeMemoryClient:
    def __init__(self):
        self.writes = []

    def search_memory(self, query, top_k=5):
        return {
            "semantic_facts": [
                {"id": 7, "statement": "The launch review is Friday."}
            ]
        }

    def write_memory(self, statement):
        self.writes.append(statement)
        return {"id": 8, "statement": statement, "confidence": 0.85}


class _FailingSearchClient:
    def search(self, query, *, cache=None):
        return []

    def health(self):
        return {"available": False, "last_error": "network unavailable"}


def test_calculate_allows_arithmetic_only():
    assert calculate("(12 + 8) * 2") == 40

    try:
        calculate("__import__('os').getcwd()")
    except ValueError as exc:
        assert "arithmetic" in str(exc) or "numeric" in str(exc)
    else:
        raise AssertionError("unsafe expression was accepted")

    try:
        calculate("(-1) ** 0.5")
    except ValueError as exc:
        assert "real number" in str(exc)
    else:
        raise AssertionError("complex result was accepted")


def test_search_result_is_structured_and_source_aware():
    runner = ActionRunner(search_client=_FakeSearchClient())
    action = Action.create(ActionType.SEARCH, "amtavla")

    result = runner.run(action, user_input="tell me about amtavla", search_cache={})

    assert result.ok is True
    assert isinstance(result.output[0], SearchResult)
    assert result.source_ids == [result.output[0].source_id]
    assert result.to_dict()["output"][0]["title"] == "Amtavla architecture"


def test_memory_actions_return_structured_results():
    memory = _FakeMemoryClient()
    runner = ActionRunner(search_client=_FakeSearchClient(), memory_client=memory)

    search_result = runner.run(
        Action.create(ActionType.MEMORY_SEARCH, "launch review"),
        user_input="when is it?",
    )
    write_result = runner.run(
        Action.create(ActionType.MEMORY_WRITE, "The launch review is Friday."),
        user_input="remember it",
    )

    assert search_result.ok is True
    assert search_result.source_ids == ["memory:semantic:7"]
    assert write_result.ok is True
    assert write_result.output["id"] == 8
    assert memory.writes == ["The launch review is Friday."]


def test_memory_write_requires_explicit_user_authorization():
    memory = _FakeMemoryClient()
    runner = ActionRunner(search_client=_FakeSearchClient(), memory_client=memory)

    result = runner.run(
        Action.create(ActionType.MEMORY_WRITE, "The launch review is Friday."),
        user_input="When is the launch review?",
    )

    assert result.ok is False
    assert "explicit user request" in result.error
    assert memory.writes == []


def test_action_failure_is_data_not_an_exception():
    runner = ActionRunner(search_client=_FakeSearchClient())

    result = runner.run(
        Action.create(ActionType.CALCULATE, "2 ** 999"),
        user_input="calculate it",
    )

    assert result.ok is False
    assert result.output is None
    assert "Exponent" in result.error
    assert result.duration_ms >= 0


def test_search_provider_failure_is_not_reported_as_success():
    runner = ActionRunner(search_client=_FailingSearchClient())

    result = runner.run(
        Action.create(ActionType.SEARCH, "current facts"),
        user_input="look this up",
    )

    assert result.ok is False
    assert result.output is None
    assert result.error == "Search failed: network unavailable"
