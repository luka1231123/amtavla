import re

from brain.ltm_tree import _safe_chat

MODEL = "qwen2.5-coder:1.5b"

PLANNER_PROMPT = """You are a planning assistant. Create a short todo list to answer the user's question.

Available tools:
- weather: for weather questions (Tokyo, London)
- bash: for listing files, python version, date, user, disk space, pwd

Rules:
- Max 5 steps total
- Use EXACTLY these formats, one per line:
  SEARCH: short query
  TOOL: weather
  TOOL: bash
  THINK

- SEARCH: write a short search query (max 8 words)
- TOOL: use weather or bash only if relevant
- THINK: always include for reasoning
- Always include at least 1 SEARCH step

Examples:
User: What is Python?
SEARCH: Python programming language overview
TOOL: bash
THINK

User: Weather in Tokyo?
SEARCH: current weather Tokyo
TOOL: weather
THINK

User: How do decorators work?
SEARCH: Python decorators tutorial
THINK

User: {user_input}
TODO:
"""


def generate_plan(user_input: str, context: str) -> list[tuple[str, str]]:
    prompt = PLANNER_PROMPT.format(user_input=user_input)
    raw = _safe_chat([{"role": "user", "content": prompt}], model=MODEL)
    print(f"   [DEBUG-PLANNER] -> Raw plan:\n{raw}")

    steps = []
    for line in raw.splitlines():
        line = line.strip()
        line = re.sub(r"^[\d\.\-\*\#]+\s*", "", line)
        line = line.strip()
        if not line:
            continue
        upper = line.upper()

        if upper.startswith("SEARCH"):
            query = line.split(":", 1)[1].strip() if ":" in line else line[6:].strip()
            query = query.strip('"').strip("'").strip("`").strip("*")
            query = query[:60].strip()
            if query:
                steps.append(("SEARCH", query))
        elif upper.startswith("TOOL"):
            tool_part = (
                line.split(":", 1)[1].strip() if ":" in line else line[4:].strip()
            )
            tool_lower = tool_part.lower()
            if "weather" in tool_lower:
                steps.append(("TOOL", "weather"))
            elif "bash" in tool_lower:
                steps.append(("TOOL", "bash"))
        elif upper.startswith("THINK"):
            steps.append(("THINK", ""))

    seen = set()
    deduped = []
    for step in steps:
        if step not in seen:
            seen.add(step)
            deduped.append(step)
    steps = deduped[:5]

    if not steps:
        steps = [("SEARCH", user_input[:60]), ("THINK", "")]
    elif not any(s[0] == "SEARCH" for s in steps):
        steps.insert(0, ("SEARCH", user_input[:60]))

    print(f"   [DEBUG-PLANNER] -> Parsed steps: {steps}")
    return steps
