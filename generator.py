import llama_client
from brain.prompt_builder import PromptBuilder

PROMPT_BUILDER = PromptBuilder()


def generate_response(
    user_prompt,
    plan,
    plan_results,
    context_blocks,
    intent: str | None = None,
    pathway: str | None = None,
):
    client = llama_client

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

    if context_blocks.get("semantic_facts"):
        lines = [
            f"- {item.get('statement', '')}"
            for item in context_blocks["semantic_facts"]
        ]
        context_parts.append("--- Semantic Memory ---\n" + "\n".join(lines))

    if context_blocks.get("episodic_context"):
        lines = [
            f"- User: {item.get('user_input', '')} | Bot: {item.get('response', '')}"
            for item in context_blocks["episodic_context"]
        ]
        context_parts.append("--- Episodic Recall ---\n" + "\n".join(lines))

    if context_blocks.get("combined_context"):
        context_parts.append(
            "--- Recall Engine Context ---\n" + context_blocks["combined_context"]
        )

    if context_blocks.get("ltm_context"):
        ltm = context_blocks["ltm_context"]
        if isinstance(ltm, list):
            lines = [f"- {item.get('thesis', '')}" for item in ltm]
            context_parts.append("--- Long-Term Insights ---\n" + "\n".join(lines))
        else:
            context_parts.append("--- Long-Term Insights ---\n" + str(ltm))

    if context_blocks.get("web_context"):
        context_parts.append("--- Web Grounding ---\n" + context_blocks["web_context"])

    context_str = "\n\n".join(context_parts)
    system_message = PROMPT_BUILDER.build_generator_prompt(
        assembled_context=context_str,
        intent=intent,
        pathway=pathway,
    )

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]

    try:
        response = client.chat(messages)
        return response["message"]["content"].strip()
    except Exception as e:
        return f"I apologize, but I encountered an error generating a response: {e}"
