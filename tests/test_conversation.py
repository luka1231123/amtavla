#!/usr/bin/env python3
"""
Fast conversation simulation test.
Runs a single realistic multi-turn conversation with the agent and logs everything.

Run: python3 tests/test_conversation.py
"""

import json
import os
import sys
import tempfile
import time
from collections import deque
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brain.memory_controller import MemoryController
from brain.stm import read_stm, clear_stm
from router import classify_intent
from generator import generate_response
from tools import tool_weather, tool_bash_simulator


CONVERSATION = [
    ("Hey", "chat"),
    ("What is Python and why is it so popular?", "chat"),
    ("Can you show me how decorators work?", "chat"),
    ("What about classes and inheritance?", "chat"),
    ("How do I list all files in the current directory?", "bash"),
    ("What's the weather in Tokyo?", "weather"),
    ("Interesting. How does React compare to Vue for frontend?", "chat"),
    ("What are Python context managers?", "chat"),
    ("Thanks, that's all", "chat"),
]


def run_conversation(turns, stm_file, tree_file):
    log = {
        "timestamp": datetime.now().isoformat(),
        "turns": [],
        "summary": {},
    }

    memory = MemoryController(stm_file=stm_file, tree_file=tree_file)
    history = deque(maxlen=5)
    start = time.time()

    for i, (user_input, expected_intent) in enumerate(turns):
        turn_start = time.time()
        turn_log = {
            "turn": i + 1,
            "user_input": user_input,
            "expected_intent": expected_intent,
        }

        context = memory.get_context_for_prompt(user_input)
        turn_log["stm_context"] = context["working_memory"]
        turn_log["ltm_context"] = context["ltm_context"]

        intent = classify_intent(
            user_input, context["working_memory"] + context["ltm_context"]
        )
        turn_log["actual_intent"] = intent
        turn_log["intent_match"] = (
            intent == expected_intent.upper()
            if expected_intent != "chat"
            else intent in ("CHAT", "BASH", "WEATHER")
        )

        if intent == "WEATHER":
            tool_output = tool_weather(user_input, context["working_memory"])
        elif intent == "BASH":
            tool_output = tool_bash_simulator(user_input, context["working_memory"])
        else:
            tool_output = "No tool used. Just chat."
        turn_log["tool_output"] = tool_output

        response = generate_response(
            user_input,
            list(history),
            context,
            tool_output,
        )
        turn_log["response"] = response
        turn_log["duration"] = round(time.time() - turn_start, 2)

        history.append((user_input, response))
        memory.process_turn_async(user_input, response)

        log["turns"].append(turn_log)

    if memory._memory_thread and memory._memory_thread.is_alive():
        memory._memory_thread.join()

    total_time = round(time.time() - start, 2)
    memory.tree.save()

    branches = memory.tree._collect_all_nodes()
    log["summary"] = {
        "total_turns": len(turns),
        "total_time_seconds": total_time,
        "avg_turn_time": round(total_time / len(turns), 2),
        "branches_created": len(branches),
        "branch_topics": [n["topic"] for n in branches],
        "branch_details": [
            {
                "topic": n["topic"],
                "content_count": len(n["content"]),
                "children": len(n["children"]),
            }
            for n in branches
        ],
        "final_stm": read_stm(stm_file=stm_file),
        "tree_visualization": memory.tree.visualize(),
    }

    intent_results = [t["actual_intent"] for t in log["turns"]]
    log["summary"]["intent_sequence"] = intent_results

    return log


def main():
    print("=" * 70)
    print("  amtavla Conversation Simulation Test")
    print("=" * 70)
    print()

    stm_file = tempfile.mktemp(suffix=".txt")
    tree_file = tempfile.mktemp(suffix=".json")

    print(f"Running {len(CONVERSATION)}-turn conversation...")
    print()

    log = run_conversation(CONVERSATION, stm_file, tree_file)

    print("--- Turn-by-Turn Log ---")
    print()
    for turn in log["turns"]:
        print(f"Turn {turn['turn']}: {turn['user_input']}")
        print(
            f"  Intent: {turn['actual_intent']} (expected: {turn['expected_intent']})"
        )
        print(f"  Tool: {turn['tool_output'][:80]}...")
        print(f"  Response: {turn['response'][:120]}...")
        print(f"  Duration: {turn['duration']}s")
        print()

    print("--- Summary ---")
    s = log["summary"]
    print(f"Total turns: {s['total_turns']}")
    print(f"Total time: {s['total_time_seconds']}s")
    print(f"Avg turn time: {s['avg_turn_time']}s")
    print(f"Branches created: {s['branches_created']}")
    print(f"Branch topics: {s['branch_topics']}")
    print(f"Intent sequence: {s['intent_sequence']}")
    print()

    print("--- Tree Visualization ---")
    print(s["tree_visualization"])
    print()

    print("--- Final STM ---")
    print(s["final_stm"] or "(empty)")
    print()

    log_path = os.path.join(os.path.dirname(__file__), "conversation_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Full log saved to: {log_path}")
    print()

    cleanup = [stm_file, tree_file]
    for p in cleanup:
        if os.path.exists(p):
            os.remove(p)

    print("=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
