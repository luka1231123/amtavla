import json
import re

from brain.ltm_tree import _safe_chat

LLAMA_CLIENT = None

def _get_client():
    global LLAMA_CLIENT
    if LLAMA_CLIENT is None:
        import llama_client
        LLAMA_CLIENT = llama_client
    return LLAMA_CLIENT

PLANNER_PROMPT = """You are a planning assistant. Create a todo list to answer the user's question.

Rules:
- Max 5 steps total
- Always include at least 1 SEARCH step
- Output ONLY valid JSON, no other text

JSON format:
{{
  "steps": [
    {{"action": "SEARCH", "detail": "short query"}},
    {{"action": "TOOL", "detail": "bash"}},
    {{"action": "THINK", "detail": ""}}
  ],
  "thinking": "your reasoning about how to approach this question"
}}

Context from memory:
{context}

Examples:
User: What is Python?
{{"steps": [{{"action": "SEARCH", "detail": "Python programming language"}}, {{"action": "THINK", "detail": ""}}], "thinking": "Need to search for basic info about Python"}}

User: How do decorators work?
{{"steps": [{{"action": "SEARCH", "detail": "Python decorators tutorial"}}, {{"action": "THINK", "detail": ""}}], "thinking": "Search for Python decorator concepts"}}

User: {user_input}
Output only JSON:"""


def _parse_plan(raw: str) -> list[tuple[str, str]]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start:end])
            except json.JSONDecodeError:
                return []
        else:
            return []

    steps = []
    for step in data.get("steps", []):
        action = step.get("action", "").upper()
        detail = step.get("detail", "")
        if action in ("SEARCH", "TOOL", "THINK"):
            steps.append((action, detail))

    deduped = []
    seen = set()
    for step in steps:
        if step not in seen:
            seen.add(step)
            deduped.append(step)

    deduped = deduped[:5]

    if not deduped:
        deduped = [("SEARCH", ""), ("THINK", "")]
    elif not any(s[0] == "SEARCH" for s in deduped):
        deduped.insert(0, ("SEARCH", ""))

    return deduped


def generate_plan(user_input: str, context: str) -> tuple[list[tuple[str, str]], str]:
    client = _get_client()
    context_part = context[:500] if context else "No prior context."
    prompt = PLANNER_PROMPT.format(user_input=user_input, context=context_part)

    thinking = ""
    try:
        response = client.chat([{"role": "user", "content": prompt}])
        raw = response.get("message", {}).get("content", "")
        thinking = get_thinking(raw)
    except Exception:
        raw = ""

    if not raw:
        return [("SEARCH", user_input[:60]), ("THINK", "")], thinking

    return _parse_plan(raw), thinking


def get_thinking(raw: str) -> str:
    try:
        data = json.loads(raw)
        return data.get("thinking", "")
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                data = json.loads(raw[start:end])
                return data.get("thinking", "")
            except json.JSONDecodeError:
                pass
    return ""
