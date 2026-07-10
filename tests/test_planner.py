from brain.contracts import ActionType
from brain.planner import Planner, _parse_plan, parse_plan


def test_parse_plan_accepts_supported_actions():
    raw = '{"steps":[{"action":"SEARCH","detail":"python"},{"action":"THINK","detail":""}]}'
    assert _parse_plan(raw) == [("SEARCH", "python"), ("THINK", "")]


def test_parse_plan_reports_unsupported_action_when_no_valid_steps():
    raw = '{"steps":[{"action":"TOOL","detail":"ls"}]}'
    assert _parse_plan(raw) == [("THINK", "Unsupported planner action skipped: TOOL")]


def test_parse_plan_accepts_phase_one_actions_and_reports_mixed_unknowns():
    raw = """
    {"steps":[
      {"action":"CALCULATE","detail":"2 + 2"},
      {"action":"MEMORY_SEARCH","detail":"launch date"},
      {"action":"MEMORY_WRITE","detail":"launch date is Friday"},
      {"action":"SHELL","detail":"rm -rf /"}
    ]}
    """

    plan = parse_plan(raw)

    assert [item.action_type for item in plan.actions] == [
        ActionType.CALCULATE,
        ActionType.MEMORY_SEARCH,
        ActionType.MEMORY_WRITE,
    ]
    assert plan.warnings == ["Unsupported planner action: SHELL"]


def test_parse_plan_dedupes_and_limits_steps():
    raw = """
    {"steps":[
      {"action":"THINK","detail":"a"},
      {"action":"THINK","detail":"a"},
      {"action":"THINK","detail":"b"},
      {"action":"THINK","detail":"c"},
      {"action":"THINK","detail":"d"},
      {"action":"THINK","detail":"e"},
      {"action":"THINK","detail":"f"}
    ]}
    """
    plan = parse_plan(raw)
    assert plan.to_pairs() == [
        ("THINK", "a"),
        ("THINK", "b"),
        ("THINK", "c"),
        ("THINK", "d"),
        ("THINK", "e"),
    ]
    assert plan.warnings == ["Planner output truncated to 5 actions"]


class _FakePlannerClient:
    def chat(self, messages):
        assert "Allowed actions" in messages[0]["content"]
        return {
            "message": {
                "content": '{"steps":[{"action":"CALCULATE","detail":"6*7"}],"thinking":"exact"}'
            }
        }


def test_planner_supports_injected_client_without_local_model():
    plan = Planner(client=_FakePlannerClient()).create_plan("six times seven", "")

    assert plan.to_pairs() == [("CALCULATE", "6*7")]
    assert plan.thinking == "exact"
