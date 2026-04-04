import sys

sys.path.insert(0, ".")

from router import classify_intent
from generator import generate_final_response
from tools import tool_weather, tool_bash_simulator
from brain.memory_controller import MemoryController

MEMORY = MemoryController()


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
        combined_memory = (
            f"Working Memory:\n{context['working_memory']}\n{context['ltm_context']}"
        )

        intent = classify_intent(user_input, combined_memory)
        context_data = ""

        if intent == "WEATHER":
            context_data = tool_weather(user_input, combined_memory)
        elif intent == "BASH":
            context_data = tool_bash_simulator(user_input, combined_memory)
        else:
            context_data = "No tool used. Just chat."

        response = generate_final_response(user_input, context_data)
        print(f"{response}\n")

        MEMORY.process_turn(user_input, response)


if __name__ == "__main__":
    run()
