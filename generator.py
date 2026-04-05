import ollama

MODEL = "qwen2.5-coder:1.5b"

SYSTEM_PROMPT = """You are amtavla, a CLI assistant.

You have access to short-term memory (STM), long-term memory (LTM), web search results, and tools.
Use the context provided to inform your responses.
Be concise and direct. No yapping.
Cite sources from web search when using them."""


def generate_response(user_prompt, plan, plan_results, context_blocks):
    context_parts = []

    if plan:
        plan_lines = [f"- {action}: {detail}" for action, detail in plan]
        context_parts.append("--- Plan ---\n" + "\n".join(plan_lines))

    if plan_results:
        for action, detail, result in plan_results:
            if result and result.strip():
                label = action
                if detail:
                    label += f" ({detail})"
                context_parts.append(f"--- {label} ---\n{result}")

    if context_blocks.get("working_memory"):
        context_parts.append(
            "--- Short-Term Memory ---\n" + context_blocks["working_memory"]
        )

    if context_blocks.get("ltm_context"):
        context_parts.append(
            "--- Long-Term Memory ---\n" + context_blocks["ltm_context"]
        )

    context_str = "\n\n".join(context_parts)
    system_message = SYSTEM_PROMPT
    if context_str:
        system_message += f"\n\n{context_str}"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]

    response = ollama.chat(model=MODEL, messages=messages)
    return response["message"]["content"].strip()
