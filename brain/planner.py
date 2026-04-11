from brain.json_utils import extract_first_json_object
from brain.prompt_builder import PromptBuilder

LLAMA_CLIENT = None
PROMPT_BUILDER = PromptBuilder()


def _get_client():
    global LLAMA_CLIENT
    if LLAMA_CLIENT is None:
        import llama_client

        LLAMA_CLIENT = llama_client
    return LLAMA_CLIENT


def _parse_plan(raw: str) -> list[tuple[str, str]]:
    data = extract_first_json_object(raw)
    if not data:
        return []

    steps = []
    for step in data.get("steps", []):
        action = step.get("action", "").upper()
        detail = step.get("detail", "")
        if action in ("SEARCH", "THINK"):
            steps.append((action, detail))

    deduped = []
    seen = set()
    for step in steps:
        if step not in seen:
            seen.add(step)
            deduped.append(step)

    deduped = deduped[:5]

    if not deduped:
        deduped = [("THINK", "")]

    return deduped


def generate_plan(
    user_input: str,
    context: str,
    intent: str | None = None,
    pathway: str | None = None,
) -> tuple[list[tuple[str, str]], str]:
    client = _get_client()
    prompt = PROMPT_BUILDER.build_planner_prompt(
        user_input=user_input,
        memory_context=context,
        intent=intent,
        pathway=pathway,
    )

    thinking = ""
    try:
        response = client.chat([{"role": "user", "content": prompt}])
        raw = response.get("message", {}).get("content", "")
        thinking = get_thinking(raw)
    except Exception:
        raw = ""

    if not raw:
        return [("THINK", "")], thinking

    return _parse_plan(raw), thinking


def get_thinking(raw: str) -> str:
    data = extract_first_json_object(raw)
    if not data:
        return ""
    value = data.get("thinking", "")
    return value if isinstance(value, str) else ""
