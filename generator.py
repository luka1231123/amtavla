import ollama

MODEL = "qwen2.5-coder:1.5b"

SYSTEM_PROMPT = """You are amtavla, a CLI assistant.

You have access to short-term memory (STM), long-term memory (LTM), and tools.
Use the context provided to inform your responses.
Be concise and direct. No yapping."""


def generate_response(user_prompt, conversation_history, context_blocks, tool_output):
    context_parts = []

    if conversation_history:
        history_lines = []
        for user_msg, assistant_msg in conversation_history:
            history_lines.append(f"User: {user_msg}")
            history_lines.append(f"Assistant: {assistant_msg}")
        context_parts.append("--- Recent Conversation ---\n" + "\n".join(history_lines))

    if context_blocks.get("working_memory"):
        context_parts.append(
            "--- Short-Term Memory ---\n" + context_blocks["working_memory"]
        )

    if context_blocks.get("ltm_context"):
        context_parts.append(
            "--- Long-Term Memory ---\n" + context_blocks["ltm_context"]
        )

    if tool_output:
        context_parts.append("--- Tool Output ---\n" + tool_output)

    context_str = "\n\n".join(context_parts)
    system_message = SYSTEM_PROMPT
    if context_str:
        system_message += f"\n\n{context_str}"

    messages = [{"role": "system", "content": system_message}]
    for user_msg, assistant_msg in conversation_history:
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": user_prompt})

    response = ollama.chat(model=MODEL, messages=messages)
    return response["message"]["content"].strip()
