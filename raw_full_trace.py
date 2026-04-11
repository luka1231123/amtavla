#!/usr/bin/env python3

import json
import os
import sqlite3
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

import llama_client
import tools.websearch as websearch_module
from brain.config import load_brain_config
from brain.intent_router import IntentRouter
from brain.memory_controller import MemoryController
from brain.memory import service as memory_service_module
from brain.planner import generate_plan
from generator import generate_response


PROMPTS = [
    "hi there",
    "I like concise answers with bullet points.",
    "Remember this: my name is Mira.",
    "Remember this: my bike is blue and parked in garage B2.",
    "What do you remember about me so far?",
    "Give me a 3-step morning routine for focus.",
    "What are two differences between RAM and storage?",
    "What is 17 * 19?",
    "Who wrote Pride and Prejudice?",
    "What is the capital of Japan?",
    "Can you explain recursion simply?",
    "Give one Python example of recursion.",
    "I have a meeting Tuesday 14:00, remind me tomorrow morning.",
    "Based on my notes, make a compact checklist.",
    "Where is my bike parked?",
    "What color is my bike?",
    "What is my name?",
    "Tell me one memory trivia question about what I told you.",
    "Ask me a quick quiz from our conversation.",
    "Now answer that quiz yourself briefly.",
    "List files in current directory and explain the command.",
    "/brain status",
    "/brain full",
    "/ask",
    "/idle",
    "Summarize everything in 5 bullets.",
    "If I forget, where is my bike?",
    "/delete",
    "What do you remember about my bike now?",
    "/brain status",
    "thanks",
    "exit",
]


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _indent(text: str, spaces: int = 2) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line for line in text.splitlines())


def _render_obj(value, level: int = 0) -> list[str]:
    lead = "  " * level
    if isinstance(value, dict):
        if not value:
            return [lead + "(empty)"]
        out = []
        for key, val in value.items():
            if isinstance(val, (dict, list)):
                out.append(f"{lead}{key}:")
                out.extend(_render_obj(val, level + 1))
            else:
                out.append(f"{lead}{key}: {val}")
        return out
    if isinstance(value, list):
        if not value:
            return [lead + "(empty)"]
        out = []
        for idx, item in enumerate(value, 1):
            if isinstance(item, (dict, list)):
                out.append(f"{lead}- item {idx}:")
                out.extend(_render_obj(item, level + 1))
            else:
                out.append(f"{lead}- {item}")
        return out
    return [lead + str(value)]


def render(value) -> str:
    return "\n".join(_render_obj(value))


def parse_json_text(text: str):
    if not text or not isinstance(text, str):
        return text
    stripped = text.strip()
    if not stripped:
        return text
    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        try:
            return json.loads(stripped)
        except Exception:
            return text
    return text


class TraceWriter:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.files = {
            "brain": (run_dir / "brain_timeline.txt").open("w", encoding="utf-8"),
            "session": (run_dir / "session_transcript.txt").open("w", encoding="utf-8"),
            "memory": (run_dir / "memory_chronological.txt").open(
                "w", encoding="utf-8"
            ),
            "web": (run_dir / "web_searches.txt").open("w", encoding="utf-8"),
            "idle": (run_dir / "idle_processes.txt").open("w", encoding="utf-8"),
            "thoughts": (run_dir / "thoughts.txt").open("w", encoding="utf-8"),
        }

    def line(self, channel: str, message: str):
        stamped = f"[{ts()}] {message}\n"
        f = self.files[channel]
        f.write(stamped)
        f.flush()

    def block(self, channel: str, title: str, body: str):
        self.line(channel, title)
        if body:
            for line in body.splitlines():
                self.line(channel, f"  {line}")

    def terminal(self, message: str):
        print(message)

    def close(self):
        for f in self.files.values():
            try:
                f.close()
            except Exception:
                pass


class MemoryChronicle:
    def __init__(self, memory_service, writer: TraceWriter):
        self.mem = memory_service
        self.writer = writer
        self.last_ids = {
            "events": 0,
            "recall_log": 0,
            "facts": 0,
            "insights": 0,
            "jobs": 0,
        }

    def _rows_since(self, db_path: str, table: str, last_id: int):
        with sqlite3.connect(db_path, timeout=30) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"SELECT * FROM {table} WHERE id > ? ORDER BY id ASC", (last_id,)
            ).fetchall()
        return rows

    def _max_id(self, db_path: str, table: str):
        with sqlite3.connect(db_path, timeout=30) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                f"SELECT COALESCE(MAX(id), 0) AS m FROM {table}"
            ).fetchone()
        return int(row["m"]) if row else 0

    def flush_new(self, label: str):
        self.writer.line("memory", f"=== Memory flush: {label} ===")
        table_map = {
            "events": self.mem._episodic_db,
            "recall_log": self.mem._episodic_db,
            "facts": self.mem._semantic_db,
            "insights": self.mem._insight_db,
            "jobs": self.mem._jobs_db,
        }
        for table, db_path in table_map.items():
            rows = self._rows_since(db_path, table, self.last_ids[table])
            if not rows:
                continue
            for row in rows:
                item = dict(row)
                for key in (
                    "todo_json",
                    "context_json",
                    "provenance_json",
                    "evidence_json",
                    "payload_json",
                ):
                    if key in item:
                        item[key] = parse_json_text(item[key])
                self.writer.block(
                    "memory", f"{table} id={item.get('id')}", render(item)
                )
            self.last_ids[table] = max(int(r["id"]) for r in rows)

    def mark_current_as_seen(self):
        table_map = {
            "events": (self.mem._episodic_db, "events"),
            "recall_log": (self.mem._episodic_db, "recall_log"),
            "facts": (self.mem._semantic_db, "facts"),
            "insights": (self.mem._insight_db, "insights"),
            "jobs": (self.mem._jobs_db, "jobs"),
        }
        for key, (db_path, table) in table_map.items():
            self.last_ids[key] = self._max_id(db_path, table)


def format_route(route: dict) -> str:
    return (
        f"intent={route.get('intent')} pathway={route.get('pathway')} "
        f"score={route.get('score')} confidence={route.get('confidence')} source={route.get('source')}"
    )


def execute_plan_step(action: str, detail: str, user_input: str, memory_text: str):
    del memory_text
    if action == "SEARCH":
        query = detail.strip() or user_input
        result = websearch_module.tool_websearch(query)
        return action, query, result
    return action, detail, ""


def main():
    started = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("logs") / "raw_runs" / started
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = TraceWriter(run_dir)
    writer.terminal(f"Raw full trace started: {run_dir}")
    writer.line("brain", "=== RAW FULL TRACE START ===")

    cfg = load_brain_config()
    router = IntentRouter(cfg)
    memory = MemoryController()
    chronicle = MemoryChronicle(memory.memory, writer)

    original_tool_websearch = websearch_module.tool_websearch
    original_mem_tool_websearch = memory_service_module.tool_websearch
    original_run_idle_jobs = memory.memory.run_idle_jobs

    def wrapped_tool_websearch(query: str) -> str:
        t0 = time.time()
        writer.line("web", f"tool_websearch query: {query}")
        out = original_tool_websearch(query)
        ms = int((time.time() - t0) * 1000)
        writer.block("web", f"tool_websearch result ({ms} ms)", out)
        return out

    def wrapped_run_idle_jobs():
        t0 = time.time()
        writer.line("idle", "run_idle_jobs start")
        out = original_run_idle_jobs()
        ms = int((time.time() - t0) * 1000)
        writer.block("idle", f"run_idle_jobs done ({ms} ms)", render(out))
        writer.block("brain", "idle metrics", render(out))
        return out

    websearch_module.tool_websearch = wrapped_tool_websearch
    memory_service_module.tool_websearch = wrapped_tool_websearch
    memory.memory.run_idle_jobs = wrapped_run_idle_jobs

    try:
        writer.terminal("Starting llama-server...")
        llama_client._ensure_server_running()
        writer.terminal("llama-server ready.")

        writer.terminal("Clearing memory before run...")
        memory.clear_all_memory()
        chronicle.mark_current_as_seen()
        writer.line("brain", "Memory cleared at start")

        max_steps = int(cfg.get("routing", {}).get("max_plan_steps", 4))

        for turn_idx, prompt in enumerate(PROMPTS, 1):
            turn_start = time.time()
            writer.terminal(f"\n[{turn_idx:02d}] > {prompt}")
            writer.line("session", f"TURN {turn_idx} USER: {prompt}")
            writer.line("brain", f"turn={turn_idx} user_prompt={prompt}")
            memory.begin_foreground_turn()

            lower = prompt.strip().lower()
            if lower in {"exit", "quit", "q"}:
                writer.terminal("Goodbye.")
                writer.line("session", "assistant: Goodbye.")
                memory.end_foreground_turn()
                break

            if prompt.startswith("/brain"):
                parts = prompt.split()
                mode = parts[1] if len(parts) > 1 else "status"
                out = memory.get_debug_info(mode)
                writer.terminal(out)
                writer.block("session", f"TURN {turn_idx} ASSISTANT", out)
                writer.block("brain", f"command /brain {mode}", out)
                chronicle.flush_new(f"after /brain turn {turn_idx}")
                memory.end_foreground_turn()
                continue

            if prompt.startswith("/ask"):
                forced = memory.force_proactive_ask()
                msg = (
                    forced.get("prompt")
                    or "No pending insight is ready for proactive ask."
                )
                if forced.get("insight_id"):
                    msg += f" [insight_id={forced['insight_id']}]"
                writer.terminal(msg)
                writer.block("session", f"TURN {turn_idx} ASSISTANT", msg)
                writer.block("brain", "command /ask", render(forced))
                chronicle.flush_new(f"after /ask turn {turn_idx}")
                memory.end_foreground_turn()
                continue

            if prompt.startswith("/idle"):
                out = memory.run_idle_now()
                msg = "Idle maintenance run complete.\n" + render(out)
                writer.terminal(msg)
                writer.block("session", f"TURN {turn_idx} ASSISTANT", msg)
                writer.block("brain", "command /idle", render(out))
                chronicle.flush_new(f"after /idle turn {turn_idx}")
                memory.end_foreground_turn()
                continue

            if prompt.startswith("/delete"):
                memory.clear_all_memory()
                msg = "All memory databases cleared (episodic, semantic, insight)."
                writer.terminal(msg)
                writer.block("session", f"TURN {turn_idx} ASSISTANT", msg)
                writer.block("brain", "command /delete", msg)
                chronicle.flush_new(f"after /delete turn {turn_idx}")
                memory.end_foreground_turn()
                continue

            memory.note_user_activity()
            route = router.route(prompt)
            writer.line("brain", f"route: {format_route(route)}")

            include_web = route.get("pathway") not in {
                "remember_reply",
                "memory_recall_reply",
                "brain_dump_reply",
                "direct_reply",
            }
            if route.get("intent") in {"smalltalk", "greeting"}:
                include_web = False
            if len(prompt.split()) <= 2:
                include_web = False

            context = memory.get_context_for_prompt(
                prompt,
                include_web=include_web,
                intent=route.get("intent", ""),
                pathway=route.get("pathway", ""),
            )
            context_text = context.get("combined_context", "")
            writer.block(
                "brain",
                "context summary",
                render(
                    {
                        "include_web": include_web,
                        "semantic_count": len(context.get("semantic_facts", [])),
                        "episodic_count": len(context.get("episodic_context", [])),
                        "insight_count": len(context.get("ltm_context", [])),
                        "pending_feedback_prompt": context.get(
                            "pending_feedback_prompt", ""
                        ),
                    }
                ),
            )

            pathway = route.get("pathway")
            if pathway == "search_then_reply":
                plan = [("SEARCH", prompt)]
                thinking = ""
            elif pathway in {
                "direct_reply",
                "creative_reply",
                "remember_reply",
                "memory_recall_reply",
            }:
                plan = []
                thinking = ""
            else:
                plan, thinking = generate_plan(
                    prompt,
                    context_text,
                    intent=route.get("intent"),
                    pathway=pathway,
                )
                if not plan:
                    plan = [("THINK", "")]

            filtered = []
            for action, detail in plan:
                if action == "THINK":
                    continue
                filtered.append((action, detail))
                if len(filtered) >= max_steps:
                    break
            if not filtered and plan:
                filtered = [("THINK", "")]

            writer.block(
                "brain",
                "planner",
                render(
                    {
                        "thinking": thinking,
                        "plan": [{"action": a, "detail": d} for a, d in filtered],
                    }
                ),
            )
            if thinking.strip():
                writer.block(
                    "thoughts",
                    f"TURN {turn_idx} planner thinking",
                    thinking,
                )

            plan_results = []
            for action, detail in filtered:
                t0 = time.time()
                a, d, r = execute_plan_step(action, detail, prompt, context_text)
                plan_results.append((a, d, r))
                ms = int((time.time() - t0) * 1000)
                writer.block(
                    "brain",
                    f"plan step {a} detail={d} ({ms} ms)",
                    r,
                )

            response = generate_response(
                prompt,
                filtered,
                plan_results,
                context,
                intent=route.get("intent"),
                pathway=pathway,
            )
            memory_response = response
            if context.get("pending_feedback_prompt"):
                response = response + "\n\n" + context["pending_feedback_prompt"]

            writer.terminal(_indent(response, 0))
            writer.block("session", f"TURN {turn_idx} ASSISTANT", response)
            writer.block(
                "brain",
                "response",
                render(
                    {
                        "response_preview": response[:1600],
                        "latency_ms": int((time.time() - turn_start) * 1000),
                    }
                ),
            )

            todo_list = [
                {
                    "task_id": f"turn-{turn_idx}-task-{idx + 1}",
                    "action": a,
                    "detail": d,
                }
                for idx, (a, d) in enumerate(filtered)
            ]
            memory.process_turn_async(
                prompt,
                memory_response,
                trace={
                    "intent": route.get("intent", ""),
                    "pathway": pathway or "",
                    "todo": todo_list,
                    "context": {
                        "semantic": context.get("semantic_facts", []),
                        "insights": context.get("ltm_context", []),
                        "web": context.get("web_context", ""),
                    },
                    "session_id": "raw_full_trace",
                },
            )
            memory.wait_for_idle(timeout=30.0)
            chronicle.flush_new(f"after turn {turn_idx}")

            if turn_idx % 4 == 0:
                idle_out = memory.run_idle_now()
                writer.block(
                    "idle", f"periodic idle run after turn {turn_idx}", render(idle_out)
                )
                chronicle.flush_new(f"after periodic idle {turn_idx}")

            memory.end_foreground_turn()

        final_dump = memory.get_brain_dump("full", limit=200)
        writer.block("brain", "FINAL BRAIN DUMP", final_dump)
        writer.block("session", "FINAL BRAIN DUMP", final_dump)
        chronicle.flush_new("final")
        writer.terminal(f"\nRaw trace complete. Logs saved in {run_dir}")
    except Exception as exc:
        writer.terminal(f"FATAL ERROR: {exc}")
        writer.block("brain", "fatal", traceback.format_exc())
        raise
    finally:
        websearch_module.tool_websearch = original_tool_websearch
        memory_service_module.tool_websearch = original_mem_tool_websearch
        memory.memory.run_idle_jobs = original_run_idle_jobs
        try:
            memory._shutdown_on_exit()
        except Exception:
            pass
        writer.close()


if __name__ == "__main__":
    main()
