from __future__ import annotations

from typing import Any

from brain.contracts import Action, ActionType, Plan
from brain.json_utils import extract_first_json_object
from brain.prompt_builder import PromptBuilder

LLAMA_CLIENT = None
PROMPT_BUILDER = PromptBuilder()
SUPPORTED_ACTIONS = {item.value for item in ActionType}
MAX_PLAN_STEPS = 5


def _get_client():
    global LLAMA_CLIENT
    if LLAMA_CLIENT is None:
        import llama_client

        LLAMA_CLIENT = llama_client
    return LLAMA_CLIENT


def parse_plan(raw: str, max_steps: int = MAX_PLAN_STEPS) -> Plan:
    data = extract_first_json_object(raw)
    if not data:
        return Plan(warnings=["Planner returned invalid JSON"])

    raw_steps = data.get("steps", [])
    if not isinstance(raw_steps, list):
        return Plan(warnings=["Planner steps must be a list"])

    actions = []
    warnings = []
    seen = set()
    truncated = False
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            warnings.append("Planner emitted a non-object step")
            continue
        raw_action = str(raw_step.get("action") or "").strip().upper()
        action_type = ActionType.parse(raw_action)
        if action_type is None:
            if raw_action:
                warnings.append(f"Unsupported planner action: {raw_action}")
            else:
                warnings.append("Planner emitted an action without a type")
            continue
        detail = str(raw_step.get("detail") or "").strip()
        key = (action_type.value, detail)
        if key in seen:
            continue
        seen.add(key)
        if len(actions) >= max(1, max_steps):
            truncated = True
            continue
        actions.append(Action(action_type=action_type, detail=detail))

    if truncated:
        warnings.append(f"Planner output truncated to {max(1, max_steps)} actions")

    thinking = data.get("thinking", "")
    thinking = thinking if isinstance(thinking, str) else ""

    if not actions:
        detail = ""
        unsupported = [
            warning.removeprefix("Unsupported planner action: ")
            for warning in warnings
            if warning.startswith("Unsupported planner action: ")
        ]
        if unsupported:
            detail = f"Unsupported planner action skipped: {', '.join(unsupported)}"
        actions = [Action(action_type=ActionType.THINK, detail=detail)]

    return Plan(actions=actions, thinking=thinking, warnings=warnings)


def _parse_plan(raw: str) -> list[tuple[str, str]]:
    if not extract_first_json_object(raw):
        return []
    return parse_plan(raw).to_pairs()


class Planner:
    def __init__(
        self,
        client: Any | None = None,
        prompt_builder: PromptBuilder | None = None,
        max_steps: int = MAX_PLAN_STEPS,
    ) -> None:
        self.client = client
        self.prompt_builder = prompt_builder or PROMPT_BUILDER
        self.max_steps = max(1, min(MAX_PLAN_STEPS, max_steps))

    def create_plan(
        self,
        user_input: str,
        context: str,
        intent: str | None = None,
        pathway: str | None = None,
    ) -> Plan:
        prompt = self.prompt_builder.build_planner_prompt(
            user_input=user_input,
            memory_context=context,
            intent=intent,
            pathway=pathway,
        )

        try:
            client = self.client or _get_client()
            response = client.chat([{"role": "user", "content": prompt}])
            raw = response.get("message", {}).get("content", "")
        except Exception as exc:
            return Plan(
                actions=[Action(action_type=ActionType.THINK)],
                warnings=[f"Planner call failed: {exc}"],
            )

        plan = parse_plan(raw, max_steps=self.max_steps)
        if not plan.actions:
            plan.actions = [Action(action_type=ActionType.THINK)]
        return plan


def generate_structured_plan(
    user_input: str,
    context: str,
    intent: str | None = None,
    pathway: str | None = None,
) -> Plan:
    return Planner().create_plan(user_input, context, intent, pathway)


def generate_plan(
    user_input: str,
    context: str,
    intent: str | None = None,
    pathway: str | None = None,
) -> tuple[list[tuple[str, str]], str]:
    plan = generate_structured_plan(user_input, context, intent, pathway)
    return plan.to_pairs(), plan.thinking


def get_thinking(raw: str) -> str:
    data = extract_first_json_object(raw)
    if not data:
        return ""
    value = data.get("thinking", "")
    return value if isinstance(value, str) else ""
