from brain.contracts import (
    Action,
    ActionResult,
    ActionType,
    ContextPack,
    Plan,
    RouteDecision,
    SearchResult,
    utc_now,
)
from generator import ResponseGenerator, build_response_context


class _FakeClient:
    def __init__(self):
        self.messages = None

    def chat(self, messages):
        self.messages = messages
        return {"message": {"content": "Grounded answer [memory:semantic:3]."}}


def test_response_contract_exposes_memory_and_action_source_ids():
    context = ContextPack.from_memory(
        {
            "semantic_facts": [
                {"id": 3, "statement": "The launch review is Friday."}
            ]
        }
    )
    search = SearchResult.from_row(
        {
            "title": "Release calendar",
            "url": "https://example.test/calendar",
            "snippet": "Friday review confirmed.",
        },
        query="launch review",
        rank=1,
    )
    action = Action.create(ActionType.SEARCH, "launch review")
    result = ActionResult(
        action_id=action.action_id,
        action_type=action.action_type,
        detail=action.detail,
        ok=True,
        output=[search],
        sources=[search.to_source()],
        completed_at=utc_now(),
    )
    rendered = build_response_context(Plan([action]), [result], context)

    assert "[memory:semantic:3]" in rendered
    assert f"[{search.source_id}]" in rendered
    assert "Source Catalog" in rendered


def test_response_generator_uses_injected_client():
    client = _FakeClient()
    generator = ResponseGenerator(client=client)
    context = ContextPack.from_memory(
        {"semantic_facts": [{"id": 3, "statement": "Friday review."}]}
    )

    response = generator.generate(
        "When is the review?",
        Plan(),
        [],
        context,
        RouteDecision("memory_recall", "memory_recall_reply"),
    )

    assert response == "Grounded answer [memory:semantic:3]."
    assert "memory:semantic:3" in client.messages[0]["content"]


def test_recent_conversation_is_replayed_as_chat_messages():
    # Follow-ups like "what is the continuation of that phrase" only resolve if
    # the model can see the immediately-preceding exchange as real dialogue.
    client = _FakeClient()
    generator = ResponseGenerator(client=client)
    context = ContextPack.from_memory(
        {
            "conversation": [
                {
                    "user_input": "to err is human",
                    "response": "That is a common idiom about mistakes.",
                },
            ]
        }
    )

    generator.generate(
        "what is the continuation of that phrase",
        Plan(),
        [],
        context,
        RouteDecision("web_factual", "search_then_reply"),
    )

    roles = [m["role"] for m in client.messages]
    # system, prior user, prior assistant, current user — in order.
    assert roles == ["system", "user", "assistant", "user"]
    assert client.messages[1]["content"] == "to err is human"
    assert client.messages[3]["content"] == "what is the continuation of that phrase"


def test_unified_memory_item_replaces_duplicate_legacy_prompt_context():
    context = ContextPack.from_memory(
        {
            "memory_items": [
                {
                    "id": 12,
                    "item_type": "fact",
                    "content": "Sarah's birthday is May 13",
                    "review_state": "corrected",
                    "metadata": {"legacy_fact_id": 3},
                }
            ],
            "semantic_facts": [
                {"id": 3, "statement": "Sarah's birthday is May 12"}
            ],
        }
    )

    rendered = build_response_context(Plan(), [], context)

    assert "[memory:item:12]" in rendered
    assert "May 13" in rendered
    assert "May 12" not in rendered


def test_failed_action_renders_loud_failure_instruction():
    action = Action.create(ActionType.REMINDER, "remind me friday")
    result = ActionResult(
        action_id=action.action_id,
        action_type=action.action_type,
        detail=action.detail,
        ok=False,
        output=None,
        error="A reminder needs an exact time",
        completed_at=utc_now(),
    )
    rendered = build_response_context(
        Plan([action]), [result], ContextPack.from_memory({})
    )

    assert "ACTION FAILED" in rendered
    assert "did NOT happen" in rendered
    assert "Do not claim or imply success" in rendered
