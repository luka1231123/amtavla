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
        self.reminders = []
        self.research_topics = []

    def search_memory(self, query, top_k=5):
        return {
            "semantic_facts": [
                {"id": 7, "statement": "The launch review is Friday."}
            ]
        }

    def write_memory(self, statement):
        self.writes.append(statement)
        return {"id": 8, "statement": statement, "confidence": 0.85}

    def recent_notes(self, limit=20):
        return [
            {"id": 3, "item_type": "fact", "content": "Buy milk."},
            {"id": 5, "item_type": "commitment", "content": "Call the dentist."},
        ]

    def create_reminder(self, content, *, due_at=None):
        self.reminders.append((content, due_at))
        return {"id": 9, "content": content}

    def queue_research(self, topic):
        self.research_topics.append(topic)
        return {"job_id": 4, "topic": topic, "status": "pending"}


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


def test_calculate_normalizes_natural_language():
    assert calculate("what's 5 plus 3") == 8
    assert calculate("15% of 200") == 30
    assert calculate("What is 17 * 19?") == 323
    assert calculate("240 divided by 6") == 40


def test_calculate_falls_back_to_user_input_via_normalization():
    runner = ActionRunner(search_client=_FakeSearchClient())
    # Empty detail -> falls back to raw user_input, which normalization rescues.
    result = runner.run(
        Action.create(ActionType.CALCULATE, ""),
        user_input="What is 17 * 19?",
    )
    assert result.ok is True
    assert result.output["value"] == 323


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


def test_summarize_returns_note_material_with_sources():
    memory = _FakeMemoryClient()
    runner = ActionRunner(search_client=_FakeSearchClient(), memory_client=memory)

    result = runner.run(
        Action.create(ActionType.SUMMARIZE, "checklist from recent notes"),
        user_input="make a checklist from my notes",
    )

    assert result.ok is True
    assert "[memory:item:3] (fact) Buy milk." in result.output["material"]
    assert result.source_ids == ["memory:item:3", "memory:item:5"]
    assert "Summarize ONLY" in result.output["instruction"]


def test_reminder_requires_explicit_request_and_stores_due_time():
    memory = _FakeMemoryClient()
    runner = ActionRunner(search_client=_FakeSearchClient(), memory_client=memory)

    denied = runner.run(
        Action.create(ActionType.REMINDER, "call the dentist tomorrow"),
        user_input="I should call the dentist sometime",
    )
    assert denied.ok is False
    assert "explicit user request" in denied.error
    assert memory.reminders == []

    granted = runner.run(
        Action.create(ActionType.REMINDER, "remind me to call the dentist in 2 hours"),
        user_input="remind me to call the dentist in 2 hours",
    )
    assert granted.ok is True
    assert granted.output["content"] == "call the dentist in 2 hours"
    assert granted.output["due_at"] is not None
    assert granted.source_ids == ["memory:item:9"]
    content, due_at = memory.reminders[0]
    assert content == "call the dentist in 2 hours"
    assert due_at is not None


def test_reminder_without_time_is_not_stored():
    memory = _FakeMemoryClient()
    runner = ActionRunner(search_client=_FakeSearchClient(), memory_client=memory)

    result = runner.run(
        Action.create(ActionType.REMINDER, "remind me about the visa paperwork"),
        user_input="remind me about the visa paperwork",
    )

    assert result.ok is False
    assert "needs an exact time" in result.error
    assert memory.reminders == []


def test_reminder_accepts_self_declared_commitment_with_deadline():
    memory = _FakeMemoryClient()
    runner = ActionRunner(search_client=_FakeSearchClient(), memory_client=memory)

    result = runner.run(
        Action.create(ActionType.REMINDER, "I promised Nino I'll send the invoice by Friday"),
        user_input="I promised Nino I'll send the invoice by Friday.",
    )

    assert result.ok is True
    assert "invoice" in result.output["content"]
    assert result.output["due_at"] is not None
    assert len(memory.reminders) == 1


def test_reminder_source_excerpt_uses_stored_content():
    class _LockedMemory(_FakeMemoryClient):
        def create_reminder(self, content, *, due_at=None):
            # Simulate upsert matching a locked item: keep the stored content
            # while returning the correct id.
            self.reminders.append((content, due_at))
            return {"id": 9, "content": "existing locked reminder text"}

    memory = _LockedMemory()
    runner = ActionRunner(search_client=_FakeSearchClient(), memory_client=memory)

    result = runner.run(
        Action.create(ActionType.REMINDER, "remind me to call the dentist in 2 hours"),
        user_input="remind me to call the dentist in 2 hours",
    )

    assert result.ok is True
    assert result.sources[0].excerpt == "existing locked reminder text"
    assert result.sources[0].title == "Reminder: existing locked reminder text"


def test_note_read_is_sandboxed_to_root(tmp_path):
    from tools.localfiles import LocalFilesClient

    (tmp_path / "notes.md").write_text("groceries: milk", encoding="utf-8")
    runner = ActionRunner(
        search_client=_FakeSearchClient(),
        files_client=LocalFilesClient(root=tmp_path),
    )

    read = runner.run(
        Action.create(ActionType.NOTE_READ, "read notes.md"),
        user_input="read notes.md",
    )
    assert read.ok is True
    assert read.output["content"] == "groceries: milk"
    assert len(read.source_ids) == 1 and read.source_ids[0].startswith("file:")

    escape = runner.run(
        Action.create(ActionType.NOTE_READ, "read ../outside.txt"),
        user_input="read ../outside.txt",
    )
    assert "outside allowed root" in escape.output["error"]
    assert escape.source_ids == []


def test_clarify_requires_a_question_and_returns_it():
    runner = ActionRunner(search_client=_FakeSearchClient())

    empty = runner.run(
        Action.create(ActionType.CLARIFY, ""),
        user_input="fix it",
    )
    assert empty.ok is False

    asked = runner.run(
        Action.create(ActionType.CLARIFY, "What should I fix?"),
        user_input="fix it",
    )
    assert asked.ok is True
    assert asked.output == {"question": "What should I fix?"}


def test_research_queues_background_job():
    memory = _FakeMemoryClient()
    runner = ActionRunner(search_client=_FakeSearchClient(), memory_client=memory)

    result = runner.run(
        Action.create(ActionType.RESEARCH, "local-first sync engines"),
        user_input="do a deep dive on local-first sync engines",
    )

    assert result.ok is True
    assert result.output["status"] == "queued"
    assert result.output["job_id"] == 4
    assert memory.research_topics == ["local-first sync engines"]


def test_search_provider_failure_is_not_reported_as_success():
    runner = ActionRunner(search_client=_FailingSearchClient())

    result = runner.run(
        Action.create(ActionType.SEARCH, "current facts"),
        user_input="look this up",
    )

    assert result.ok is False
    assert result.output is None
    assert result.error == "Search failed: network unavailable"
