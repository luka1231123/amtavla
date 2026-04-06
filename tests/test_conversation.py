#!/usr/bin/env python3
"""
Conversation simulation test.
Runs a single realistic multi-turn conversation with the agent and logs everything.

Run: python3 tests/test_conversation.py
"""

import json
import logging
import os
import sys
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "conversation_test.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("test.conversation")

from brain.memory_controller import MemoryController
from brain.planner import generate_plan
from brain.stm import read_stm
from generator import generate_response
from tools import tool_weather, tool_bash_simulator
from tools.websearch import tool_websearch


CONVERSATION = [
    "Hey",
    "What is Python and why is it so popular?",
    "Can you show me how decorators work?",
    "What about classes and inheritance?",
    "How do I list all files in the current directory?",
    "What's the weather in Tokyo?",
    "Interesting. How does React compare to Vue for frontend?",
    "What are Python context managers?",
    "Thanks, that's all",
]


def execute_plan_step(action, detail, user_input, memory_str):
    if action == "SEARCH":
        return action, detail, tool_websearch(detail)
    elif action == "TOOL":
        if detail == "weather":
            return action, detail, tool_weather(user_input, memory_str)
        elif detail == "bash":
            return action, detail, tool_bash_simulator(user_input, memory_str)
        else:
            return action, detail, f"Unknown tool: {detail}"
    elif action == "THINK":
        return action, detail, ""
    return action, detail, ""


def run_conversation(turns, stm_file, tree_file):
    log = {
        "timestamp": datetime.now().isoformat(),
        "turns": [],
        "summary": {},
    }

    memory = MemoryController(stm_file=stm_file, tree_file=tree_file)
    history = deque(maxlen=5)
    start = time.time()

    for i, user_input in enumerate(turns):
        turn_start = time.time()
        turn_log = {
            "turn": i + 1,
            "user_input": user_input,
        }

        context = memory.get_context_for_prompt(user_input)
        turn_log["stm_context"] = context["working_memory"]
        turn_log["ltm_context"] = context["ltm_context"]

        context_text = (
            (context.get("working_memory", "") or "")
            + " "
            + (context.get("ltm_context", "") or "")
        )

        plan = generate_plan(user_input, context_text)
        turn_log["plan"] = [(a, d) for a, d in plan]

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
                    plan_results.append(
                        (result_action, result_detail, result[:300] if result else "")
                    )
                except Exception as e:
                    plan_results.append((action, detail, f"Error: {e}"))

        turn_log["plan_results"] = [(a, d, r[:100]) for a, d, r in plan_results]

        response = generate_response(user_input, plan, plan_results, context)
        turn_log["response"] = response
        turn_log["duration"] = round(time.time() - turn_start, 2)

        history.append((user_input, response))
        memory.process_turn_async(user_input, response)

        log["turns"].append(turn_log)

    time.sleep(0.5)
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
        plan_str = ", ".join(f"{a}: {d}" for a, d in turn["plan"])
        print(f"  Plan: {plan_str}")
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
    print()

    print("--- Tree Visualization ---")
    print(s["tree_visualization"])
    print()

    print("--- Final STM ---")
    print(s["final_stm"] or "(empty)")
    print()

    log_path = os.path.join(LOG_DIR, "conversation_log.json")
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
