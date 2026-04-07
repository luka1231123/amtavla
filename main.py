import json
import logging
import queue
import re
import select
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, ".")

import llama_client
from brain.config import load_brain_config
from brain.intent_router import IntentRouter
from brain.memory_controller import MemoryController
from brain.planner import generate_plan
from generator import generate_response
from logging_setup import configure_logging
from tools import tool_bash_simulator
from tools.websearch import tool_websearch

configure_logging()
logger = logging.getLogger("amtavla.main")
CONFIG = load_brain_config()
MEMORY = MemoryController()
ROUTER = IntentRouter(CONFIG)

command_queue = queue.Queue()
stop_command_poller = False


def _poll_commands():
    global stop_command_poller
    while not stop_command_poller:
        try:
            req = urllib.request.Request("http://127.0.0.1:8081/command")
            with urllib.request.urlopen(req, timeout=2) as response:
                data = response.read()
                resp = json.loads(data)
                if resp.get("command"):
                    ack_req = urllib.request.Request(
                        "http://127.0.0.1:8081/command/ack",
                        data=b"{}",
                        headers={"Content-Type": "application/json"},
                    )
                    urllib.request.urlopen(ack_req, timeout=2)
                    command_queue.put(resp["command"])
        except Exception:
            logger.debug("Command poll failed", exc_info=True)
        time.sleep(0.5)


def _send_response_to_ui(response: str):
    try:
        data = json.dumps({"text": response}).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:8081/response",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        logger.debug("Failed to send response to UI", exc_info=True)


def _clip_text(value, max_chars: int = 3000):
    if value is None:
        return ""
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"


def _send_debug_event(event_type: str, payload: dict):
    summary = ""
    if event_type == "user_prompt":
        summary = f"User prompt: {_clip_text(payload.get('prompt', ''), 100)}"
    elif event_type == "intent_decision":
        summary = (
            f"Intent={payload.get('intent')} pathway={payload.get('pathway')} "
            f"confidence={payload.get('confidence', 0):.2f}"
        )
    elif event_type == "route_pathway":
        summary = f"Routing pathway: {payload.get('pathway')} ({payload.get('intent')})"
    elif event_type == "context":
        summary = "Recall context prepared (semantic+episodic+insights+web)."
    elif event_type == "plan":
        summary = f"Plan built with {len(payload.get('steps', []))} steps."
    elif event_type == "plan_result":
        summary = f"Executed {payload.get('action')} step."
    elif event_type == "assistant_response":
        summary = "Assistant response generated."

    envelope = {
        "summary": summary,
        "data": payload,
    }
    try:
        body = json.dumps({"type": event_type, "payload": envelope}).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:8081/debug/event",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=2)
    except Exception:
        logger.debug("Failed to send debug event", exc_info=True)


def _compress_search_result(query: str, raw: str) -> str:
    max_chars = int(CONFIG.get("routing", {}).get("max_search_chars", 2200))
    if not raw:
        return ""

    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
    if not blocks:
        return _clip_text(raw, max_chars)

    query_tokens = set(re.findall(r"[a-z0-9]+", query.lower()))
    ranked = []
    for block in blocks:
        lower = block.lower()
        score = 0
        for token in query_tokens:
            if token and token in lower:
                score += 1
        ranked.append((score, block))

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = [item[1] for item in ranked[:3]]

    lines = []
    for idx, block in enumerate(selected, 1):
        block_lines = [line.strip() for line in block.splitlines() if line.strip()]
        title = block_lines[0] if block_lines else f"Result {idx}"
        url = block_lines[1] if len(block_lines) > 1 else ""
        snippet = block_lines[2] if len(block_lines) > 2 else ""
        line = f"[{idx}] {title}"
        if url:
            line += f"\n    {url}"
        if snippet:
            line += f"\n    {snippet[:260]}"
        lines.append(line)

    compressed = "\n\n".join(lines)
    return _clip_text(compressed, max_chars)


def execute_plan_step(
    action: str, detail: str, user_input: str, memory_text: str
) -> tuple[str, str, str]:
    if action == "SEARCH":
        query = detail.strip() or user_input
        raw = tool_websearch(query)
        result = _compress_search_result(query, raw)
        return action, query, result
    if action == "TOOL":
        if detail == "bash":
            result = tool_bash_simulator(user_input, memory_text)
        else:
            result = f"Unknown tool: {detail}"
        return action, detail, result
    if action == "THINK":
        return action, detail, ""
    return action, detail, ""


def _run_plan_steps(plan: list[tuple[str, str]], user_input: str, context_text: str):
    plan_results = [None] * len(plan)
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(
                execute_plan_step, action, detail, user_input, context_text
            ): (idx, action, detail)
            for idx, (action, detail) in enumerate(plan)
        }
        for future in as_completed(futures):
            idx, action, detail = futures[future]
            try:
                plan_results[idx] = future.result()
            except Exception as e:
                plan_results[idx] = (action, detail, f"Error: {e}")

    return [r for r in plan_results if r is not None]


def _route_to_plan(user_input: str, route: dict, context_text: str):
    pathway = route.get("pathway", "planner_full")
    thinking = ""
    if pathway in (
        "direct_reply",
        "creative_reply",
        "remember_reply",
        "memory_recall_reply",
    ):
        return [], [], thinking

    if pathway == "search_then_reply":
        plan = [("SEARCH", user_input)]
        return plan, _run_plan_steps(plan, user_input, context_text), thinking

    if pathway == "tool_then_reply":
        plan = [("TOOL", "bash")]
        return plan, _run_plan_steps(plan, user_input, context_text), thinking

    plan, thinking = generate_plan(user_input, context_text)
    if not plan:
        plan = [("THINK", "")]

    filtered = []
    max_steps = int(CONFIG.get("routing", {}).get("max_plan_steps", 4))
    for action, detail in plan:
        if action == "THINK":
            continue
        filtered.append((action, detail))
        if len(filtered) >= max_steps:
            break

    if not filtered:
        filtered = [("THINK", "")]
    return filtered, _run_plan_steps(filtered, user_input, context_text), thinking


def _build_todo_from_plan(plan: list[tuple[str, str]]) -> list[dict]:
    todos = []
    for idx, (action, detail) in enumerate(plan, 1):
        todos.append(
            {
                "task_id": f"task-{idx}",
                "action": action,
                "detail": detail,
                "priority": "high" if action == "TOOL" else "medium",
            }
        )
    return todos


def run():
    global stop_command_poller

    MEMORY.set_debug_hook(_send_debug_event)

    try:
        llama_client._ensure_server_running()
    except Exception as e:
        print(f"[FATAL] Failed to start llama-server: {e}")
        print(
            "Please install/build llama.cpp server and ensure a .gguf model exists in ~/llama.cpp/models."
        )
        sys.exit(1)

    poller_thread = threading.Thread(target=_poll_commands, daemon=True)
    poller_thread.start()

    print(
        "amtavla - CLI assistant (type 'exit' to quit, '/brain <mode>' debug, '/ask' proactive debug, '/idle' force idle, '/research-status', '/delete')\n"
    )
    print("Or use phone UI at http://127.0.0.1:8081\n")

    while True:
        try:
            user_input = None

            if not command_queue.empty():
                try:
                    user_input = command_queue.get_nowait()
                    print(f"[PHONE] {user_input}")
                except queue.Empty:
                    user_input = None
            elif sys.platform != "win32" and select.select([sys.stdin], [], [], 0)[0]:
                user_input = sys.stdin.readline()
                if user_input:
                    user_input = user_input.strip()
            else:
                time.sleep(0.1)
                continue

            if not user_input:
                continue
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if user_input.lower() in ("exit", "quit", "q"):
            print("Goodbye.")
            break

        if user_input.startswith("/brain"):
            parts = user_input.split()
            mode = parts[1] if len(parts) > 1 else "status"
            print(MEMORY.get_debug_info(mode))
            print()
            continue

        if user_input.startswith("/ask"):
            forced = MEMORY.force_proactive_ask()
            prompt = (
                forced.get("prompt") or "No pending insight is ready for proactive ask."
            )
            if forced.get("insight_id"):
                prompt += f" [insight_id={forced['insight_id']}]"
            print(f"{prompt}\n")
            continue

        if user_input.startswith("/idle"):
            result = MEMORY.run_idle_now()
            print("Idle maintenance run complete.")
            print(f"Status: {result.get('status')}")
            print(f"Metrics: {result.get('metrics')}\n")
            continue

        if user_input.startswith("/research-status"):
            status = MEMORY.research_status(limit=10)
            counts = status.get("counts", {})
            jobs = status.get("jobs", [])
            print("=== Research Jobs ===")
            print(f"Counts: {counts}")
            if not jobs:
                print("No research jobs yet.\n")
                continue
            for job in jobs:
                print(
                    f"- #{job['job_id']} [{job['status']}] attempts={job['attempts']} query={_clip_text(job['query'], 100)}"
                )
                if job.get("error"):
                    print(f"    error: {_clip_text(job['error'], 140)}")
            print()
            continue

        if user_input.startswith("/delete"):
            MEMORY.clear_all_memory()
            print("All memory databases cleared (episodic, semantic, insight, jobs).\n")
            continue

        try:
            MEMORY.note_user_activity()
            _send_debug_event("user_prompt", {"prompt": _clip_text(user_input, 2000)})

            route = ROUTER.route(user_input)
            _send_debug_event("intent_decision", route)
            _send_debug_event(
                "route_pathway",
                {
                    "pathway": route.get("pathway"),
                    "intent": route.get("intent"),
                },
            )

            if route.get("pathway") == "brain_dump_reply":
                mode = "full"
                text = user_input.lower()
                if "semantic" in text:
                    mode = "semantic"
                elif "episodic" in text:
                    mode = "episodic"
                elif "insight" in text or "ltm" in text:
                    mode = "insights"
                elif "job" in text:
                    mode = "jobs"
                dump = MEMORY.get_brain_dump(mode=mode)
                print(f"{dump}\n")
                _send_response_to_ui(dump)
                MEMORY.process_turn_async(
                    user_input,
                    dump,
                    trace={
                        "intent": route.get("intent", ""),
                        "todo": [],
                        "context": {"brain_dump_mode": mode},
                        "session_id": "cli",
                    },
                )
                continue

            if route.get("pathway") == "research_deep_crawl":
                job_id = MEMORY.queue_research_job(
                    user_input, source="foreground_intent"
                )
                latest = MEMORY.latest_research_result()
                response = (
                    "Started deep research in background "
                    f"(job #{job_id}). I will keep crawling sources with budget limits and summarize findings."
                )
                if latest and latest.get("result"):
                    preview = _clip_text(latest.get("result", ""), 1200)
                    response += (
                        "\n\nMost recent completed deep-research result (previous job):\n"
                        + preview
                    )
                _send_debug_event(
                    "research_job",
                    {
                        "job_id": job_id,
                        "query": _clip_text(user_input, 500),
                        "latest_previous_job": latest.get("job_id") if latest else None,
                    },
                )
                print(f"{response}\n")
                _send_response_to_ui(response)
                MEMORY.process_turn_async(
                    user_input,
                    response,
                    trace={
                        "intent": route.get("intent", ""),
                        "todo": [
                            {
                                "task_id": "research-1",
                                "action": "DEEP_CRAWL",
                                "detail": user_input,
                            }
                        ],
                        "context": {"research_job_id": job_id},
                        "session_id": "cli",
                    },
                )
                continue

            include_web = route.get("pathway") not in {
                "remember_reply",
                "memory_recall_reply",
                "brain_dump_reply",
            }
            context = MEMORY.get_context_for_prompt(user_input, include_web=include_web)
            semantic_text = "\n".join(
                f"- {item.get('statement', '')}"
                for item in context.get("semantic_facts", [])
            )
            context_text = context.get("combined_context", "") or semantic_text
            _send_debug_event(
                "context",
                {
                    "semantic_facts": _clip_text(semantic_text, 6000),
                    "combined_context": _clip_text(
                        context.get("combined_context", ""), 6000
                    ),
                    "web_context": _clip_text(context.get("web_context", ""), 6000),
                },
            )

            plan, plan_results, thinking = _route_to_plan(
                user_input, route, context_text
            )
            todo_list = _build_todo_from_plan(plan)
            _send_debug_event(
                "plan",
                {
                    "thinking": _clip_text(thinking, 800),
                    "steps": [
                        {"action": action, "detail": detail} for action, detail in plan
                    ],
                    "todo": todo_list,
                },
            )

            for action, detail, result in plan_results:
                _send_debug_event(
                    "plan_result",
                    {
                        "action": action,
                        "detail": detail,
                        "result": _clip_text(result, 6000),
                    },
                )

            response = generate_response(
                user_input,
                plan,
                plan_results,
                context,
            )
            memory_response = response
            if context.get("pending_feedback_prompt"):
                response += "\n\n" + context["pending_feedback_prompt"]
            _send_debug_event(
                "assistant_response", {"response": _clip_text(response, 8000)}
            )
            print(f"{response}\n")

            _send_response_to_ui(response)
            MEMORY.process_turn_async(
                user_input,
                memory_response,
                trace={
                    "intent": route.get("intent", ""),
                    "todo": todo_list,
                    "context": {
                        "semantic": context.get("semantic_facts", []),
                        "insights": context.get("ltm_context", []),
                        "web": context.get("web_context", ""),
                    },
                    "session_id": "cli",
                },
            )

        except Exception as e:
            print(f"   [ERROR] {e}")
            print("I'm having trouble right now. Please try again.\n")

    stop_command_poller = True


if __name__ == "__main__":
    run()
