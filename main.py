import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, ".")

from brain.planner import generate_plan
from brain.memory_controller import MemoryController
from generator import generate_response
from tools import tool_weather, tool_bash_simulator
from tools.websearch import tool_websearch

MEMORY = MemoryController()
CONVERSATION_HISTORY = deque(maxlen=5)

GREETING_RESPONSES = {
    "hi": "Hey! How can I help?",
    "hello": "Hello! What can I do for you?",
    "hey": "Hey there! What's up?",
}


def execute_plan_step(
    action: str, detail: str, user_input: str, memory: str
) -> tuple[str, str, str]:
    if action == "SEARCH":
        result = tool_websearch(detail)
        return action, detail, result
    elif action == "TOOL":
        if detail == "weather":
            result = tool_weather(user_input, memory)
        elif detail == "bash":
            result = tool_bash_simulator(user_input, memory)
        else:
            result = f"Unknown tool: {detail}"
        return action, detail, result
    elif action == "THINK":
        return action, detail, ""
    return action, detail, ""


def run():
    print("amtavla - CLI assistant (type 'exit' to quit, '/brain <mode>' for debug)\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        if user_input.startswith("/brain"):
            parts = user_input.split()
            mode = parts[1] if len(parts) > 1 else "status"
            print(MEMORY.get_debug_info(mode))
            print()
            continue

        lower = user_input.lower()
        for greeting, resp in GREETING_RESPONSES.items():
            if (
                lower == greeting
                or lower.startswith(greeting + " ")
                or lower.startswith(greeting + "!")
            ):
                print(f"{resp}\n")
                MEMORY.process_turn_async(user_input, resp)
                break
        else:
            try:
                context = MEMORY.get_context_for_prompt(user_input)
                context_text = (
                    (context.get("working_memory", "") or "")
                    + " "
                    + (context.get("ltm_context", "") or "")
                )

                plan = generate_plan(user_input, context_text)

                plan_results = []
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {
                        executor.submit(
                            execute_plan_step, action, detail, user_input, context_text
                        ): (action, detail)
                        for action, detail in plan
                    }
                    for future in as_completed(futures):
                        action, detail = futures[future]
                        try:
                            result_action, result_detail, result = future.result()
                            plan_results.append((result_action, result_detail, result))
                        except Exception as e:
                            plan_results.append((action, detail, f"Error: {e}"))

                response = generate_response(
                    user_input,
                    plan,
                    plan_results,
                    context,
                )
                print(f"{response}\n")

                CONVERSATION_HISTORY.append((user_input, response))
                MEMORY.process_turn_async(user_input, response)

            except Exception as e:
                print(f"   [ERROR] {e}")
                print("I'm having trouble right now. Please try again.\n")


if __name__ == "__main__":
    run()
