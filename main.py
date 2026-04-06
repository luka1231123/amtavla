import os
import sys
import time
import threading
import select
import queue
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, ".")

from brain.planner import generate_plan
from brain.memory_controller import MemoryController
from generator import generate_response
from tools import tool_bash_simulator
from tools.websearch import tool_websearch
import llama_client

MEMORY = MemoryController()
logger = logging.getLogger("amtavla.main")

command_queue = queue.Queue()
stop_command_poller = False


def _poll_commands():
    import urllib.request
    import json

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
        import urllib.request
        import json

        data = json.dumps({"text": response}).encode("utf-8")
        req = urllib.request.Request(
            "http://127.0.0.1:8081/response",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


def execute_plan_step(
    action: str, detail: str, user_input: str, memory: str
) -> tuple[str, str, str]:
    if action == "SEARCH":
        result = tool_websearch(detail)
        return action, detail, result
    elif action == "TOOL":
        if detail == "bash":
            result = tool_bash_simulator(user_input, memory)
        else:
            result = f"Unknown tool: {detail}"
        return action, detail, result
    elif action == "THINK":
        return action, detail, ""
    return action, detail, ""


def run():
    global stop_command_poller

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

    print("amtavla - CLI assistant (type 'exit' to quit, '/brain <mode>' for debug)\n")
    print("Or use phone UI at http://127.0.0.1:8081\n")

    while True:
        try:
            user_input = None

            # Check for phone commands first
            if not command_queue.empty():
                try:
                    user_input = command_queue.get_nowait()
                    print(f"[PHONE] {user_input}")
                except queue.Empty:
                    user_input = None
            # Check for stdin input (non-blocking)
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

        try:
            context = MEMORY.get_context_for_prompt(user_input)
            context_text = (
                (context.get("working_memory", "") or "")
                + " "
                + (context.get("ltm_context", "") or "")
            )

            plan, _ = generate_plan(user_input, context_text)

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
                        result_action, result_detail, result = future.result()
                        plan_results[idx] = (result_action, result_detail, result)
                    except Exception as e:
                        plan_results[idx] = (action, detail, f"Error: {e}")

            plan_results = [r for r in plan_results if r is not None]

            response = generate_response(
                user_input,
                plan,
                plan_results,
                context,
            )
            print(f"{response}\n")

            _send_response_to_ui(response)

            MEMORY.process_turn_async(user_input, response)

        except Exception as e:
            print(f"   [ERROR] {e}")
            print("I'm having trouble right now. Please try again.\n")

    stop_command_poller = True


if __name__ == "__main__":
    run()
