"""JSON Schemas for grammar-constrained model output.

These describe the shape of every structured model call in the system. When
`llm.structured_output_enabled` is on and the runtime supports it, they are
passed to llama.cpp as an OpenAI-style `json_schema` response format, which the
server converts to a GBNF grammar — so malformed output is impossible at the
sampler rather than merely discouraged by the prompt. Callers can then drop
their "returned invalid JSON" fallback branches.

Enums are derived from the live contracts/config so a schema can never drift
out of sync with the code that consumes it.
"""

from __future__ import annotations

from brain.config import load_brain_config
from brain.contracts import ActionType

# Action verbs the planner may emit, straight from the executable enum.
ACTION_TYPES: list[str] = [action.value for action in ActionType]


def known_pathways() -> list[str]:
    """Pathways the router may route to, derived from configured intents."""
    intents = load_brain_config().get("intents", [])
    pathways = {
        str(intent.get("pathway"))
        for intent in intents
        if intent.get("pathway")
    }
    # planner_full is the default/fallback and may not be attached to any single
    # intent, so guarantee it is always selectable.
    pathways.add("planner_full")
    return sorted(pathways)


def route_schema() -> dict:
    """One classification: intent label, target pathway, confidence."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["intent", "pathway", "confidence"],
        "properties": {
            "intent": {"type": "string", "minLength": 1},
            "pathway": {"type": "string", "enum": known_pathways()},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }


def plan_schema(max_steps: int = 5) -> dict:
    """A bounded plan: short reasoning plus an ordered list of typed actions."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["actions"],
        "properties": {
            "thinking": {"type": "string"},
            "actions": {
                "type": "array",
                "maxItems": max(1, int(max_steps)),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["action"],
                    "properties": {
                        "action": {"type": "string", "enum": ACTION_TYPES},
                        "detail": {"type": "string"},
                    },
                },
            },
        },
    }


# Commit-time extraction (S2): one semantic pass replaces the regex stack.
# Time resolution stays in code — the model returns the *phrase*, never a
# timestamp, because time math must never be generative.
EXTRACTION_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["user_facts", "commitments", "preferences", "corrections"],
    "properties": {
        "user_facts": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "description": "Durable facts the user stated about themselves or their world.",
        },
        "commitments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["task"],
                "properties": {
                    "task": {"type": "string", "minLength": 1},
                    "time_phrase": {
                        "type": "string",
                        "description": "Verbatim time expression, e.g. 'in 20 minutes', 'friday'. Empty if none.",
                    },
                },
            },
        },
        "preferences": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "description": "Standing preferences ('I prefer concise answers').",
        },
        "corrections": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
            "description": "Statements that correct a previously stated fact.",
        },
    },
}


# Insight synthesis (S6): the model may decline by returning keep=false, so
# silence is a first-class outcome rather than forced concatenation.
INSIGHT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "required": ["keep", "thesis"],
    "properties": {
        "keep": {"type": "boolean"},
        "thesis": {"type": "string"},
        "rationale": {"type": "string"},
    },
}
