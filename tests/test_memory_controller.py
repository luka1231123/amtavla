#!/usr/bin/env python3

"""
System-level memory and loop test with realistic conversations.

Run:
  python3 -m unittest tests.test_memory_controller

This test intentionally avoids deep mocks. It runs the real planning + tool +
generation pipeline and writes AI-friendly logs for each turn.
"""

import json
import os
import sys
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brain.memory_controller import MemoryController
from brain.stm import read_stm
from brain.planner import generate_plan
from generator import generate_response
from tools import tool_bash_simulator, tool_weather
from tools.websearch import tool_websearch


def _execute_plan_step(action, detail, user_input, context_text):
    if action == "SEARCH":
        return action, detail, tool_websearch(detail)
    if action == "TOOL":
        if detail == "weather":
            return action, detail, tool_weather(user_input, context_text)
        if detail == "bash":
            return action, detail, tool_bash_simulator(user_input, context_text)
        return action, detail, f"Unknown tool: {detail}"
    return action, detail, ""


def _run_scenario(name, turns, logs_root):
    stm_fd, stm_file = tempfile.mkstemp(suffix=".json")
    tree_fd, tree_file = tempfile.mkstemp(suffix=".json")
    os.close(stm_fd)
    os.close(tree_fd)

    session_dir = os.path.join(logs_root, name)
    os.makedirs(session_dir, exist_ok=True)
    jsonl_path = os.path.join(session_dir, "turns.jsonl")
    session_path = os.path.join(session_dir, "session.json")

    memory = MemoryController(stm_file=stm_file, tree_file=tree_file)
    turns_log = []
    started = time.time()

    try:
        with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
            for idx, user_input in enumerate(turns, 1):
                turn_started = time.time()

                context = memory.get_context_for_prompt(user_input)
                context_text = (
                    (context.get("working_memory", "") or "")
                    + " "
                    + (context.get("ltm_context", "") or "")
                )

                plan, thinking = generate_plan(user_input, context_text)

                ordered_results = [None] * len(plan)
                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {
                        executor.submit(
                            _execute_plan_step, action, detail, user_input, context_text
                        ): (plan_idx, action, detail)
                        for plan_idx, (action, detail) in enumerate(plan)
                    }
                    for future in as_completed(futures):
                        plan_idx, action, detail = futures[future]
                        try:
                            ordered_results[plan_idx] = future.result()
                        except Exception as exc:
                            ordered_results[plan_idx] = (
                                action,
                                detail,
                                f"Error: {exc}",
                            )

                plan_results = [r for r in ordered_results if r is not None]
                response = generate_response(user_input, plan, plan_results, context)
                memory.process_turn_async(user_input, response)

                turn_log = {
                    "turn": idx,
                    "user_input": user_input,
                    "plan": [{"action": a, "detail": d} for a, d in plan],
                    "thinking": thinking,
                    "plan_results": [
                        {
                            "action": a,
                            "detail": d,
                            "result": (r[:500] if isinstance(r, str) else str(r)),
                        }
                        for a, d, r in plan_results
                    ],
                    "response": response,
                    "latency_ms": int((time.time() - turn_started) * 1000),
                }
                turns_log.append(turn_log)
                jsonl_file.write(json.dumps(turn_log) + "\n")

        memory.wait_for_idle(timeout=3.0)
        memory.tree.save()

        nodes = memory.tree._collect_all_nodes()
        summary = {
            "scenario": name,
            "turn_count": len(turns_log),
            "elapsed_seconds": round(time.time() - started, 2),
            "responses_non_empty": sum(1 for t in turns_log if t["response"].strip()),
            "search_steps": sum(
                1 for t in turns_log for p in t["plan"] if p["action"] == "SEARCH"
            ),
            "tool_steps": sum(
                1 for t in turns_log for p in t["plan"] if p["action"] == "TOOL"
            ),
            "branches_created": len(nodes),
            "branch_topics": [n["topic"] for n in nodes],
            "final_stm": read_stm(stm_file=stm_file),
            "tree_visualization": memory.tree.visualize(),
        }

        session = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "scenario": name,
            "summary": summary,
            "turns": turns_log,
        }
        with open(session_path, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2)

        return summary
    finally:
        memory._shutdown_on_exit()
        for path in [stm_file, tree_file, f"{stm_file}.lock", f"{tree_file}.lock"]:
            if os.path.exists(path):
                os.remove(path)


class MemoryControllerSystemTests(unittest.TestCase):
    def test_realistic_system_scenarios_and_logs(self):
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        logs_root = os.path.join("logs", "system_runs", timestamp)
        os.makedirs(logs_root, exist_ok=True)

        scenarios = {
            "general_knowledge_and_followups": [
                "Hey there, can we do a quick study session?",
                "Explain Python like I am a junior developer joining a backend team.",
                "Nice. Now show decorators with a tiny practical example.",
                "Cool, how is that different from a context manager?",
                "Switching gears: what is the weather in Tokyo today?",
                "Back to Python. When should I choose a class over plain functions?",
                "Thanks, give me a short recap in bullets.",
            ],
            "cli_like_commands_and_tools": [
                "Can you list files in the current directory?",
                "What Python version am I running?",
                "What is my current working directory?",
                "How much disk space do I have left?",
                "What is the date and time right now in UTC?",
                "Also, weather in London please.",
            ],
            "messy_human_language_and_topic_shift": [
                "yo quick one whats python again?",
                "ok and decorators... like in normal words not textbook",
                "btw react vs vue for a tiny side project?",
                "wait nvm go back: explain inheritance with a real world analogy",
                "uh and how do I list all files maybe with ls style",
                "last thing summarize what we covered and what I should learn next",
            ],
        }

        scenario_summaries = []
        for name, turns in scenarios.items():
            scenario_summaries.append(_run_scenario(name, turns, logs_root))

        aggregate = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "logs_root": logs_root,
            "scenario_count": len(scenario_summaries),
            "total_turns": sum(s["turn_count"] for s in scenario_summaries),
            "total_non_empty_responses": sum(
                s["responses_non_empty"] for s in scenario_summaries
            ),
            "scenario_summaries": scenario_summaries,
        }
        summary_path = os.path.join(logs_root, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(aggregate, f, indent=2)

        self.assertEqual(aggregate["scenario_count"], 3)
        self.assertGreaterEqual(aggregate["total_turns"], 18)
        self.assertEqual(
            aggregate["total_non_empty_responses"], aggregate["total_turns"]
        )


if __name__ == "__main__":
    unittest.main()
