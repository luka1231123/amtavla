#!/usr/bin/env python3

import json
import os
import sys
import time
import unittest
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from brain.config import load_brain_config
from brain.intent_router import IntentRouter
from brain.memory_controller import MemoryController
from brain.planner import generate_plan
from generator import generate_response
from tools import tool_bash_simulator
from tools.websearch import tool_websearch


PROMPTS = [
    "Hey, I prefer concise technical answers.",
    "Explain Python decorators with one practical use case.",
    "What are 2 major tradeoffs between React and Vue?",
    "List files in current directory and tell me what command does it.",
    "Remember that my project deadline is next Friday and I need a checklist.",
    "Based on what we discussed, give a concise action plan.",
]


def _run_plan(plan, user_input, memory_text):
    results = []
    for action, detail in plan:
        try:
            if action == "SEARCH":
                query = detail.strip() or user_input
                results.append((action, query, tool_websearch(query)))
            elif action == "TOOL":
                if detail == "bash":
                    results.append(
                        (action, detail, tool_bash_simulator(user_input, memory_text))
                    )
                else:
                    results.append((action, detail, f"Unknown tool: {detail}"))
            else:
                results.append((action, detail, ""))
        except Exception as exc:
            results.append((action, detail, f"Error: {exc}"))
    return results


class AbilitiesTest(unittest.TestCase):
    def test_system_abilities_end_to_end(self):
        cfg = load_brain_config()
        router = IntentRouter(cfg)
        memory = MemoryController()
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("logs", "abilities_runs", timestamp)
        os.makedirs(run_dir, exist_ok=True)

        turn_log_path = os.path.join(run_dir, "turns.jsonl")
        trace_log_path = os.path.join(run_dir, "brain_trace.jsonl")
        error_log_path = os.path.join(run_dir, "errors.jsonl")
        summary_path = os.path.join(run_dir, "summary.json")

        turns = []
        traces = []
        errors = []

        try:
            with (
                open(turn_log_path, "w", encoding="utf-8") as turn_file,
                open(trace_log_path, "w", encoding="utf-8") as trace_file,
                open(error_log_path, "w", encoding="utf-8") as error_file,
            ):
                for idx, prompt in enumerate(PROMPTS, 1):
                    started = time.time()
                    error_text = ""
                    response = ""

                    route = router.route(prompt)
                    context = memory.get_context_for_prompt(prompt)
                    context_text = context.get("combined_context", "")

                    if route.get("pathway") == "search_then_reply":
                        plan = [("SEARCH", prompt)]
                    elif route.get("pathway") == "tool_then_reply":
                        plan = [("TOOL", "bash")]
                    elif route.get("pathway") in (
                        "direct_reply",
                        "creative_reply",
                        "remember_reply",
                    ):
                        plan = []
                    else:
                        plan, _ = generate_plan(prompt, context_text)
                        if not plan:
                            plan = [("THINK", "")]

                    todo = [
                        {
                            "task_id": f"turn-{idx}-task-{pidx + 1}",
                            "action": action,
                            "detail": detail,
                        }
                        for pidx, (action, detail) in enumerate(plan)
                    ]

                    plan_results = _run_plan(plan, prompt, context_text)
                    try:
                        response = generate_response(
                            prompt, plan, plan_results, context
                        )
                        if context.get("pending_feedback_prompt"):
                            response += "\n\n" + context["pending_feedback_prompt"]
                    except Exception as exc:
                        error_text = str(exc)
                        response = f"Error while generating response: {exc}"

                    trace = {
                        "turn": idx,
                        "prompt": prompt,
                        "intent": route,
                        "semantic_facts": context.get("semantic_facts", []),
                        "recall_context_preview": context_text[:1500],
                        "todo": todo,
                        "plan": [{"action": a, "detail": d} for a, d in plan],
                        "plan_results": [
                            {
                                "action": a,
                                "detail": d,
                                "result_preview": (r or "")[:500],
                            }
                            for a, d, r in plan_results
                        ],
                        "response_preview": response[:1200],
                        "error": error_text,
                        "latency_ms": int((time.time() - started) * 1000),
                    }
                    traces.append(trace)
                    trace_file.write(json.dumps(trace) + "\n")

                    turn_entry = {
                        "turn": idx,
                        "prompt": prompt,
                        "response": response,
                        "latency_ms": trace["latency_ms"],
                    }
                    turns.append(turn_entry)
                    turn_file.write(json.dumps(turn_entry) + "\n")

                    if error_text:
                        err = {"turn": idx, "prompt": prompt, "error": error_text}
                        errors.append(err)
                        error_file.write(json.dumps(err) + "\n")

                    memory.process_turn_async(
                        prompt,
                        response,
                        trace={
                            "intent": route.get("intent", ""),
                            "todo": todo,
                            "context": {
                                "semantic": context.get("semantic_facts", []),
                                "insights": context.get("ltm_context", []),
                                "web": context.get("web_context", ""),
                            },
                            "error": error_text,
                            "session_id": "abilities_test",
                        },
                    )

            memory.wait_for_idle(timeout=6.0)
            time.sleep(float(cfg.get("idle_memory", {}).get("idle_seconds", 4.0)) + 0.2)
            status = memory.memory.get_status()

            summary = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "prompt_count": len(PROMPTS),
                "turn_count": len(turns),
                "error_count": len(errors),
                "memory_status": status,
                "run_dir": run_dir,
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            self.assertEqual(len(turns), len(PROMPTS))
            self.assertTrue(all(t["response"].strip() for t in turns))
        finally:
            memory._shutdown_on_exit()


if __name__ == "__main__":
    unittest.main()
