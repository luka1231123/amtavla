import sys
from collections import deque

sys.path.insert(0, ".")

from router import classify_intent
from generator import generate_response
from tools import tool_weather, tool_bash_simulator
from brain.memory_controller import MemoryController

MEMORY = MemoryController()
CONVERSATION_HISTORY = deque(maxlen=5)


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

        context = MEMORY.get_context_for_prompt(user_input)

        intent = classify_intent(
            user_input, context["working_memory"] + context["ltm_context"]
        )
        tool_output = ""

        if intent == "WEATHER":
            tool_output = tool_weather(user_input, context["working_memory"])
        elif intent == "BASH":
            tool_output = tool_bash_simulator(user_input, context["working_memory"])
        else:
            tool_output = "No tool used. Just chat."

        response = generate_response(
            user_input,
            list(CONVERSATION_HISTORY),
            context,
            tool_output,
        )
        print(f"{response}\n")

        CONVERSATION_HISTORY.append((user_input, response))

        MEMORY.process_turn_async(user_input, response)


if __name__ == "__main__":
    run()
