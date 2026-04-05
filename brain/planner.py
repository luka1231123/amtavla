from brain.ltm_tree import _safe_chat

MODEL = "qwen2.5-coder:1.5b"

TOOLS_LIST = """
Available tools:
- weather: for weather, temperature, climate questions (cities: Tokyo, London)
- bash: for listing files, checking python version, date/time, current user, disk space, pwd
- websearch: for any topic needing current or detailed information
"""

PLANNER_PROMPT = """You are a planning assistant. Create a short todo list of steps to fully answer the user's question.

{tools_list}

Rules:
- Max 5 steps
- Use these prefixes: SEARCH, TOOL, MEMORY, THINK
- SEARCH: a short websearch query (max 8 words)
- TOOL: which tool to use (weather or bash)
- MEMORY: retrieve relevant context from long-term memory
- THINK: reasoning step to synthesize the answer
- Always include at least 1 SEARCH step
- Keep it brief

User: {user_input}
Context: {context}

TODO:
"""


def generate_plan(user_input: str, context: str) -> list[str]:
    prompt = PLANNER_PROMPT.format(
        tools_list=TOOLS_LIST,
        user_input=user_input,
        context=context[:500] if context else "No context available",
    )
    raw = _safe_chat([{"role": "user", "content": prompt}], model=MODEL)
    print(f"   [DEBUG-PLANNER] -> Raw plan:\n{raw}")

    steps = []
    for line in raw.splitlines():
        line = line.strip().lstrip("- ").lstrip("* ").strip()
        if not line:
            continue
        upper = line.upper()
        if upper.startswith("SEARCH"):
            query = (
                line.split(":", 1)[1].strip().strip('"').strip("'")
                if ":" in line
                else line[6:].strip()
            )
            if query:
                steps.append(("SEARCH", query))
        elif upper.startswith("TOOL"):
            tool_part = (
                line.split(":", 1)[1].strip() if ":" in line else line[4:].strip()
            )
            if "weather" in tool_part.lower():
                steps.append(("TOOL", "weather"))
            elif "bash" in tool_part.lower():
                steps.append(("TOOL", "bash"))
        elif upper.startswith("MEMORY"):
            steps.append(("MEMORY", ""))
        elif upper.startswith("THINK"):
            steps.append(("THINK", ""))

    steps = steps[:5]
    print(f"   [DEBUG-PLANNER] -> Parsed steps: {steps}")
    return steps
