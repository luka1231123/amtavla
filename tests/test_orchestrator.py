import asyncio
import json

from brain.action_runner import ActionRunner
from brain.contracts import Action, ActionType, Plan, SearchResult
from brain.orchestrator import TurnOrchestrator


class _FakeRouter:
    def __init__(self, intent="web_factual", pathway="search_then_reply"):
        self.intent = intent
        self.pathway = pathway

    def route(self, text):
        return {
            "intent": self.intent,
            "pathway": self.pathway,
            "confidence": 0.9,
            "score": 3,
            "source": "fake",
        }


class _FakeMemory:
    def __init__(self, context=None):
        self.context = context or {}
        self.commits = []
        self.foreground = 0
        self.writes = []

    def begin_foreground_turn(self):
        self.foreground += 1

    def end_foreground_turn(self):
        self.foreground -= 1

    def get_context_for_prompt(
        self, user_input, include_web, intent, pathway, search_cache
    ):
        assert include_web is False
        return self.context

    def process_turn_async(self, user_input, response, trace):
        self.commits.append((user_input, response, trace))

    def get_brain_dump(self, mode):
        return f"brain dump: {mode}"

    def search_memory(self, query, top_k=5):
        return self.context

    def write_memory(self, statement):
        self.writes.append(statement)
        return {"id": 11, "statement": statement, "confidence": 0.85}


class _FakeSearch:
    def search(self, query, *, cache=None):
        return [
            SearchResult.from_row(
                {
                    "title": "Current source",
                    "url": "https://example.test/current",
                    "snippet": "Current grounded information.",
                },
                query=query,
                rank=1,
            )
        ]

    def health(self):
        return {"available": True, "last_error": ""}


class _FakePlanner:
    def __init__(self, plan=None):
        self.plan = plan or Plan()

    def create_plan(self, user_input, context, intent, pathway):
        return self.plan


class _FakeGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, user_input, plan, results, context, route):
        self.calls.append((user_input, plan, results, context, route))
        return "grounded response"


class _FakeHealth:
    def snapshot(self):
        return {
            "model": {"available": True},
            "search": {"available": True},
            "embedding": {"available": True},
        }


def _run(orchestrator, text="what is current?"):
    return asyncio.run(orchestrator.process(text, session_id="test", input_source="test"))


def test_full_search_turn_has_structured_actions_sources_trace_and_commit():
    memory = _FakeMemory(
        {
            "semantic_facts": [
                {"id": 4, "statement": "The user prefers concise answers."}
            ],
            "combined_context": "The user prefers concise answers.",
        }
    )
    generator = _FakeGenerator()
    runner = ActionRunner(search_client=_FakeSearch(), memory_client=memory)
    orchestrator = TurnOrchestrator(
        router=_FakeRouter(),
        memory=memory,
        planner=_FakePlanner(),
        action_runner=runner,
        response_generator=generator,
        health_reporter=_FakeHealth(),
        config={"routing": {"max_plan_steps": 5}},
    )

    turn = _run(orchestrator)

    assert turn.status == "completed"
    assert turn.plan.to_pairs() == [("SEARCH", "what is current?")]
    assert turn.action_results[0].ok is True
    assert turn.action_results[0].to_dict()["output"][0]["title"] == "Current source"
    # Available context and tool evidence are traced separately; the response
    # records only sources the final text actually cited.
    assert turn.response_source_ids == []
    assert [event.event_type for event in turn.trace] == [
        "user_prompt",
        "health",
        "intent_decision",
        "route_pathway",
        "context",
        "plan",
        "action_result",
        "assistant_response",
        "memory_commit",
    ]
    assert memory.foreground == 0
    assert memory.commits[0][2]["turn_id"] == turn.turn_id
    assert memory.commits[0][2]["actions"][0]["ok"] is True
    assert len(generator.calls) == 1
    json.dumps(turn.to_dict())


def test_planned_calculation_runs_without_model_search_or_embedding():
    memory = _FakeMemory()
    action = Action.create(ActionType.CALCULATE, "21 * 2")
    generator = _FakeGenerator()
    orchestrator = TurnOrchestrator(
        router=_FakeRouter("default", "planner_full"),
        memory=memory,
        planner=_FakePlanner(Plan(actions=[action])),
        action_runner=ActionRunner(search_client=_FakeSearch(), memory_client=memory),
        response_generator=generator,
        health_reporter=_FakeHealth(),
        config={"routing": {"max_plan_steps": 5}},
    )

    turn = _run(orchestrator, "calculate this")

    assert turn.status == "completed"
    assert turn.action_results[0].output == {"expression": "21 * 2", "value": 42}


def test_empty_memory_recall_returns_idk_without_generator_guessing():
    memory = _FakeMemory()
    generator = _FakeGenerator()
    orchestrator = TurnOrchestrator(
        router=_FakeRouter("memory_recall", "memory_recall_reply"),
        memory=memory,
        planner=_FakePlanner(),
        action_runner=ActionRunner(search_client=_FakeSearch(), memory_client=memory),
        response_generator=generator,
        health_reporter=_FakeHealth(),
        config={"routing": {"max_plan_steps": 5}},
    )

    turn = _run(orchestrator, "where did I park?")

    assert turn.response == "IDK"
    assert generator.calls == []
    assert memory.commits[0][1] == "IDK"


def test_clarify_result_is_the_response_verbatim_without_generation():
    memory = _FakeMemory()
    generator = _FakeGenerator()
    action = Action.create(ActionType.CLARIFY, "Which project do you mean?")
    orchestrator = TurnOrchestrator(
        router=_FakeRouter("default", "planner_full"),
        memory=memory,
        planner=_FakePlanner(Plan(actions=[action])),
        action_runner=ActionRunner(search_client=_FakeSearch(), memory_client=memory),
        response_generator=generator,
        health_reporter=_FakeHealth(),
        config={"routing": {"max_plan_steps": 5, "show_sources": True}},
    )

    turn = _run(orchestrator, "fix it")

    assert turn.status == "completed"
    assert turn.response == "Which project do you mean?"
    assert generator.calls == []
    assert "Sources:" not in turn.response


def test_reminder_pathway_plans_one_deterministic_reminder_action():
    memory = _FakeMemory()
    memory.reminders = []
    memory.create_reminder = lambda content, *, due_at=None: (
        memory.reminders.append((content, due_at)) or {"id": 21, "content": content}
    )
    generator = _FakeGenerator()
    orchestrator = TurnOrchestrator(
        router=_FakeRouter("reminder", "reminder_reply"),
        memory=memory,
        planner=_FakePlanner(),
        action_runner=ActionRunner(search_client=_FakeSearch(), memory_client=memory),
        response_generator=generator,
        health_reporter=_FakeHealth(),
        config={"routing": {"max_plan_steps": 5}},
    )

    turn = _run(orchestrator, "remind me to stretch in 1 hour")

    assert turn.plan.to_pairs() == [("REMINDER", "remind me to stretch in 1 hour")]
    assert turn.action_results[0].ok is True
    assert memory.reminders[0][0] == "stretch in 1 hour"
    assert turn.response_source_ids == []


def test_summarize_pathway_plans_one_summarize_action():
    memory = _FakeMemory()
    memory.recent_notes = lambda limit=20: [
        {"id": 3, "item_type": "fact", "content": "Buy milk."}
    ]
    generator = _FakeGenerator()
    orchestrator = TurnOrchestrator(
        router=_FakeRouter("summarize", "summarize_reply"),
        memory=memory,
        planner=_FakePlanner(),
        action_runner=ActionRunner(search_client=_FakeSearch(), memory_client=memory),
        response_generator=generator,
        health_reporter=_FakeHealth(),
        config={"routing": {"max_plan_steps": 5}},
    )

    turn = _run(orchestrator, "make a checklist from my notes")

    assert turn.plan.to_pairs() == [
        ("SUMMARIZE", "make a checklist from my notes")
    ]
    assert turn.action_results[0].ok is True
    assert turn.response_source_ids == []
    assert len(generator.calls) == 1


def test_action_errors_remain_visible_in_completed_turn():
    memory = _FakeMemory()
    action = Action.create(ActionType.CALCULATE, "2 ** 999")
    orchestrator = TurnOrchestrator(
        router=_FakeRouter("default", "planner_full"),
        memory=memory,
        planner=_FakePlanner(Plan(actions=[action])),
        action_runner=ActionRunner(search_client=_FakeSearch(), memory_client=memory),
        response_generator=_FakeGenerator(),
        health_reporter=_FakeHealth(),
        config={"routing": {"max_plan_steps": 5}},
    )

    turn = _run(orchestrator, "calculate this")

    assert turn.status == "completed"
    assert turn.action_results[0].ok is False
    assert "Exponent" in turn.action_results[0].error
    action_trace = next(event for event in turn.trace if event.event_type == "action_result")
    assert action_trace.outputs["ok"] is False


def test_fuzzy_reminder_route_falls_back_to_planner_when_gate_would_refuse():
    """An embedding/LLM route to reminder_reply for text with no reminder
    request must not schedule a REMINDER that is guaranteed to fail."""
    memory = _FakeMemory()
    fallback_plan = Plan(actions=[Action.create(ActionType.CLARIFY, "What thing?")])
    orchestrator = TurnOrchestrator(
        router=_FakeRouter("clarify", "reminder_reply"),
        memory=memory,
        planner=_FakePlanner(fallback_plan),
        action_runner=ActionRunner(search_client=_FakeSearch(), memory_client=memory),
        response_generator=_FakeGenerator(),
        health_reporter=_FakeHealth(),
        config={"routing": {"max_plan_steps": 5}},
    )

    turn = _run(orchestrator, "Can you make that thing better?")

    assert turn.plan.to_pairs() == [("CLARIFY", "What thing?")]
    assert turn.response == "What thing?"
    assert all(result.ok for result in turn.action_results)


def test_explicit_reminder_still_uses_deterministic_action():
    memory = _FakeMemory()
    memory.reminders = []
    memory.create_reminder = lambda content, *, due_at=None: (
        memory.reminders.append((content, due_at)) or {"id": 5, "content": content}
    )
    orchestrator = TurnOrchestrator(
        router=_FakeRouter("reminder", "reminder_reply"),
        memory=memory,
        planner=_FakePlanner(),
        action_runner=ActionRunner(search_client=_FakeSearch(), memory_client=memory),
        response_generator=_FakeGenerator(),
        health_reporter=_FakeHealth(),
        config={"routing": {"max_plan_steps": 5}},
    )

    turn = _run(orchestrator, "remind me to stretch in 1 hour")

    assert turn.plan.to_pairs() == [("REMINDER", "remind me to stretch in 1 hour")]
    assert turn.action_results[0].ok is True


def test_memory_check_stays_out_of_reply_and_reminder_nudge_is_appended():
    memory = _FakeMemory(
        {
            "combined_context": "",
            "pending_feedback_prompt": 'Quick memory check: should I keep this insight — "X"?',
            "pending_feedback_id": 7,
            "reminder_nudge": "Overdue: send the invoice.",
        }
    )
    orchestrator = TurnOrchestrator(
        router=_FakeRouter("default", "planner_full"),
        memory=memory,
        planner=_FakePlanner(),
        action_runner=ActionRunner(search_client=_FakeSearch(), memory_client=memory),
        response_generator=_FakeGenerator(),
        health_reporter=_FakeHealth(),
        config={"routing": {"max_plan_steps": 5}},
    )

    turn = _run(orchestrator, "plan my afternoon around the invoice work")

    assert "memory check" not in turn.response
    assert "Overdue: send the invoice." in turn.response
    # The question still travels on the turn for the transport layer to send
    # as its own yes/no message.
    assert turn.context.pending_feedback_prompt.startswith("Quick memory check")
    assert turn.context.pending_feedback_id == 7
