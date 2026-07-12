import asyncio
import logging
import sys
import time

sys.path.insert(0, ".")

import llama_client
import socketio
from brain.action_runner import ActionRunner
from brain.approvals import ApprovalCoordinator
from brain.config import load_brain_config
from brain.health import HealthReporter, render_health
from brain.intent_router import IntentRouter
from brain.memory_controller import MemoryController
from brain.orchestrator import TurnOrchestrator
from brain.planner import Planner
from generator import ResponseGenerator
from logging_setup import configure_logging

configure_logging()
logger = logging.getLogger("amtavla.main")
CONFIG = load_brain_config()
MEMORY = MemoryController()
ROUTER = IntentRouter(CONFIG)
# approvals=MEMORY makes T2 actions (e.g. SHELL_RUN) pause for explicit approval
# instead of failing closed; the coordinator runs them once the user approves.
ACTION_RUNNER = ActionRunner(memory_client=MEMORY, approvals=MEMORY)
APPROVALS = ApprovalCoordinator(MEMORY, ACTION_RUNNER)
HEALTH = HealthReporter(MEMORY, search_client=ACTION_RUNNER.search_client)
ORCHESTRATOR = TurnOrchestrator(
    router=ROUTER,
    memory=MEMORY,
    planner=Planner(
        max_steps=int(CONFIG.get("routing", {}).get("max_plan_steps", 5))
    ),
    action_runner=ACTION_RUNNER,
    response_generator=ResponseGenerator(),
    health_reporter=HEALTH,
    config=CONFIG,
)

REALTIME_URL = "http://127.0.0.1:8081"
SIO_CLIENT = None
ASYNC_LOOP = None


async def _emit_socket(event: str, payload: dict):
    if SIO_CLIENT is None or not SIO_CLIENT.connected:
        return
    try:
        await SIO_CLIENT.emit(event, payload)
    except Exception:
        logger.debug("Socket emit failed for event '%s'", event, exc_info=True)


def _dispatch_socket_emit(event: str, payload: dict):
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_emit_socket(event, payload))
    except RuntimeError:
        if ASYNC_LOOP is not None and ASYNC_LOOP.is_running():
            asyncio.run_coroutine_threadsafe(
                _emit_socket(event, payload),
                ASYNC_LOOP,
            )


def _send_response_to_ui(response: str):
    _dispatch_socket_emit("assistant_response", {"text": response})


# Proactive messages that could not reach the phone UI (socket down) wait here
# and are flushed on reconnect, so reminders/research results are never lost.
_PENDING_UI_MESSAGES: list[str] = []


def _deliver_proactive(message: str):
    """Idle-worker deliveries (due reminders, finished research) reach both surfaces."""
    print(f"\n{message}\n")
    if SIO_CLIENT is None or not SIO_CLIENT.connected:
        _PENDING_UI_MESSAGES.append(message)
        return
    _send_response_to_ui(message)


def _flush_pending_ui_messages():
    while _PENDING_UI_MESSAGES:
        _send_response_to_ui(_PENDING_UI_MESSAGES.pop(0))


def _send_memory_check(prompt: str, insight_id):
    """Deliver a keep/discard memory question as its own message.

    The phone UI renders it with yes/no buttons; the CLI explains how to answer.
    """
    if not prompt:
        return
    print(f"\n{prompt}\n(Answer yes or no in your next message.)\n")
    _dispatch_socket_emit(
        "memory_check", {"text": prompt, "insight_id": insight_id}
    )


def _surface_pending_approvals(turn) -> None:
    """After a turn, tell the user about any action waiting on their approval."""
    for result in turn.action_results:
        output = result.output if isinstance(result.output, dict) else {}
        if output.get("status") != "awaiting_approval":
            continue
        message = (
            f"⏸  Approval needed — {output.get('summary', result.action_type.value)}\n"
            f"    Approve with  /approve {output.get('approval_id')}   "
            f"or reject with  /deny {output.get('approval_id')}"
        )
        print(f"{message}\n")
        _send_response_to_ui(message)


def _render_shell_result(result_dict: dict) -> str:
    """Human-readable rendering of a SHELL_RUN action result."""
    output = result_dict.get("output") if isinstance(result_dict, dict) else None
    if not isinstance(output, dict):
        return "Command finished."
    lines = [f"$ {output.get('command', '')}", f"(exit {output.get('returncode')})"]
    if output.get("stdout"):
        lines.append(output["stdout"].rstrip())
    if output.get("stderr"):
        lines.append("[stderr]\n" + output["stderr"].rstrip())
    return "\n".join(lines).strip()


MEMORY.set_proactive_hook(_deliver_proactive)


def _clip_text(value, max_chars: int = 3000):
    if value is None:
        return ""
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"


def _build_debug_payload(event_type: str, event: dict) -> dict:
    inputs = event.get("inputs", {})
    outputs = event.get("outputs", {})
    summary = event_type.replace("_", " ").title()
    if event_type == "user_prompt":
        summary = f"User prompt: {_clip_text(inputs.get('text', ''), 100)}"
    elif event_type == "intent_decision":
        summary = (
            f"Intent={outputs.get('intent')} pathway={outputs.get('pathway')} "
            f"confidence={outputs.get('confidence', 0):.2f}"
        )
    elif event_type == "route_pathway":
        summary = (
            f"Routing pathway: {outputs.get('pathway')} ({outputs.get('intent')})"
        )
    elif event_type == "context":
        summary = (
            "Context prepared: "
            f"{outputs.get('semantic_count', 0)} facts, "
            f"{outputs.get('episodic_count', 0)} episodes, "
            f"{outputs.get('insight_count', 0)} insights"
        )
    elif event_type == "plan":
        summary = f"Plan built with {len(outputs.get('actions', []))} actions."
    elif event_type == "action_result":
        summary = (
            f"{outputs.get('action')} "
            f"{'completed' if outputs.get('ok') else 'failed'}."
        )
    elif event_type == "assistant_response":
        summary = "Assistant response generated."
    elif event_type == "health":
        summary = "Model, search, and embedding health checked."

    return {
        "type": event_type,
        "payload": {
            "summary": summary,
            "data": event,
        },
    }


def _send_debug_event(event_type: str, event: dict):
    _dispatch_socket_emit("debug_event", _build_debug_payload(event_type, event))


async def _connect_realtime(command_queue: asyncio.Queue[tuple[str, str]]):
    global SIO_CLIENT
    sio = socketio.AsyncClient(reconnection=True, logger=False, engineio_logger=False)

    @sio.event
    async def connect():
        logger.info("Connected to phone server via Socket.IO")
        _flush_pending_ui_messages()

    @sio.event
    async def disconnect():
        logger.warning("Disconnected from phone server")

    @sio.on("command_submitted")
    async def on_command_submitted(data):
        text = ((data or {}).get("text") or "").strip()
        if text:
            await command_queue.put((text, "phone"))

    @sio.on("insight_feedback")
    async def on_insight_feedback(data):
        payload = data or {}
        insight_id = payload.get("insight_id")
        keep = bool(payload.get("keep"))
        if insight_id is None:
            return
        try:
            await asyncio.to_thread(
                MEMORY.apply_insight_feedback, int(insight_id), keep
            )
            ack = "Kept that insight." if keep else "Discarded that insight."
        except Exception as exc:
            ack = f"Could not record that answer: {exc}"
        _send_response_to_ui(ack)
        print(f"\n{ack}\n")

    attempts = 0
    while attempts < 10:
        attempts += 1
        try:
            await sio.connect(REALTIME_URL, wait_timeout=2)
            SIO_CLIENT = sio
            return
        except Exception:
            logger.debug("Socket.IO connect attempt %d failed", attempts, exc_info=True)
            await asyncio.sleep(0.5)
    logger.warning("Running without phone realtime bridge (server unreachable)")


async def run():
    global ASYNC_LOOP
    ASYNC_LOOP = asyncio.get_running_loop()
    command_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

    MEMORY.set_debug_hook(_send_debug_event)
    ORCHESTRATOR.debug_hook = _send_debug_event

    try:
        await asyncio.to_thread(llama_client._ensure_server_running)
    except Exception as exc:
        print(f"[WARN] Failed to start llama-server: {exc}")
        print(
            "Amtavla is running in degraded mode. Use /health for details."
        )

    await _connect_realtime(command_queue)

    stdin_reader_installed = False
    if sys.platform != "win32":

        def _on_stdin_ready():
            line = sys.stdin.readline()
            if line:
                command_queue.put_nowait((line.strip(), "cli"))

        ASYNC_LOOP.add_reader(sys.stdin, _on_stdin_ready)
        stdin_reader_installed = True

    print(
        "amtavla - CLI assistant (type 'exit' to quit, '/brain <mode>', "
        "'/health', '/ask', '/idle', '/delete', '/brief', '/loops', "
        "'/done <id>', '/focus <min|off>', '/review', "
        "'/approvals', '/approve <id>', '/deny <id>')\n"
    )
    print("Or use phone UI at http://127.0.0.1:8081\n")

    try:
        while True:
            user_input, input_source = await command_queue.get()
            if not user_input:
                continue
            command = user_input.split(maxsplit=1)[0].lower()

            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye.")
                break

            if command == "/brain":
                parts = user_input.split()
                mode = parts[1] if len(parts) > 1 else "status"
                response = await asyncio.to_thread(MEMORY.get_debug_info, mode)
                print(f"{response}\n")
                _send_response_to_ui(response)
                continue

            if command == "/health":
                health = await asyncio.to_thread(HEALTH.snapshot)
                response = render_health(health)
                print(f"{response}\n")
                _send_response_to_ui(response)
                continue

            if command == "/ask":
                forced = await asyncio.to_thread(MEMORY.force_proactive_ask)
                if forced.get("prompt"):
                    _send_memory_check(forced["prompt"], forced.get("insight_id"))
                else:
                    response = "No pending insight is ready for proactive ask."
                    print(f"{response}\n")
                    _send_response_to_ui(response)
                continue

            if command == "/idle":
                result = await asyncio.to_thread(MEMORY.run_idle_now)
                response = (
                    "Idle maintenance run complete.\n"
                    f"Status: {result.get('status')}\n"
                    f"Metrics: {result.get('metrics')}"
                )
                print(f"{response}\n")
                _send_response_to_ui(response)
                continue

            if command == "/brief":
                response = await asyncio.to_thread(MEMORY.daily_brief)
                print(f"{response}\n")
                _send_response_to_ui(response)
                continue

            if command == "/loops":
                loops = await asyncio.to_thread(MEMORY.open_commitments)
                if loops:
                    lines = ["=== Open Loops ==="]
                    for item in loops:
                        due = item["metadata"].get("due_at")
                        due_str = (
                            f" (due {time.strftime('%Y-%m-%d', time.localtime(due))})"
                            if due
                            else ""
                        )
                        unconfirmed = (
                            item["review_state"] not in {"confirmed", "corrected"}
                            and item["confidence"] < 0.7
                            and not due
                        )
                        flag = " (unconfirmed)" if unconfirmed else ""
                        lines.append(f"#{item['id']}{due_str}{flag}: {item['content']}")
                    response = "\n".join(lines)
                else:
                    response = "No open commitments."
                print(f"{response}\n")
                _send_response_to_ui(response)
                continue

            if command == "/done":
                parts = user_input.split()
                try:
                    item = await asyncio.to_thread(
                        MEMORY.complete_commitment, int(parts[1])
                    )
                    response = f"Marked done: {item['content']}"
                except (IndexError, ValueError, KeyError) as exc:
                    response = f"Usage: /done <commitment id> ({exc})"
                print(f"{response}\n")
                _send_response_to_ui(response)
                continue

            if command == "/focus":
                parts = user_input.split()
                arg = parts[1] if len(parts) > 1 else "25"
                if arg.lower() in ("off", "end", "stop"):
                    await asyncio.to_thread(MEMORY.end_focus)
                    response = "Focus session ended."
                else:
                    try:
                        minutes = float(arg)
                    except ValueError:
                        minutes = 25.0
                    until = await asyncio.to_thread(MEMORY.start_focus, minutes)
                    response = (
                        f"Focus session until "
                        f"{time.strftime('%H:%M', time.localtime(until))}. "
                        "Non-urgent prompts are suppressed."
                    )
                print(f"{response}\n")
                _send_response_to_ui(response)
                continue

            if command == "/review":
                response = await asyncio.to_thread(MEMORY.review_drill)
                print(f"{response}\n")
                _send_response_to_ui(response)
                continue

            if command == "/approvals":
                pending = await asyncio.to_thread(APPROVALS.pending)
                if pending:
                    lines = ["=== Pending approvals ==="]
                    for item in pending:
                        lines.append(f"#{item['id']}: {item['summary']}")
                    lines.append("Approve with /approve <id> or reject with /deny <id>.")
                    response = "\n".join(lines)
                else:
                    response = "No actions are waiting for approval."
                print(f"{response}\n")
                _send_response_to_ui(response)
                continue

            if command in ("/approve", "/deny"):
                parts = user_input.split()
                if len(parts) < 2 or not parts[1].lstrip("#").isdigit():
                    response = f"Usage: {command} <approval-id> (see /approvals)."
                else:
                    approval_id = int(parts[1].lstrip("#"))
                    approved = command == "/approve"
                    try:
                        outcome = await asyncio.to_thread(
                            APPROVALS.resolve, approval_id, approved
                        )
                    except KeyError:
                        outcome = None
                    if outcome is None:
                        response = f"No approval #{approval_id}."
                    elif not approved:
                        response = f"Denied. '{outcome['summary']}' will not run."
                    elif not outcome["executed"]:
                        response = (
                            f"Approval #{approval_id} was already {outcome['state']}; "
                            "nothing ran."
                        )
                    else:
                        action_result = outcome["result"]
                        if action_result.action_type.value == "SHELL_RUN":
                            body = _render_shell_result(action_result.to_dict())
                        else:
                            body = f"Done: {outcome['summary']}"
                        status = "✓" if action_result.ok else "✗ failed"
                        response = f"[{status}] {body}"
                print(f"{response}\n")
                _send_response_to_ui(response)
                continue

            if command == "/delete":
                await asyncio.to_thread(MEMORY.clear_all_memory)
                response = (
                    "All memory cleared (episodic, semantic, insight, catalog, "
                    "jobs, and vectors)."
                )
                print(f"{response}\n")
                _send_response_to_ui(response)
                continue

            turn = await ORCHESTRATOR.process(
                user_input,
                session_id=input_source,
                input_source=input_source,
            )
            print(f"{turn.response}\n")
            _send_response_to_ui(turn.response)
            _surface_pending_approvals(turn)
            if turn.context is not None and turn.context.pending_feedback_prompt:
                _send_memory_check(
                    turn.context.pending_feedback_prompt,
                    turn.context.pending_feedback_id,
                )
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
    finally:
        if stdin_reader_installed and ASYNC_LOOP is not None:
            ASYNC_LOOP.remove_reader(sys.stdin)
        if SIO_CLIENT is not None and SIO_CLIENT.connected:
            await SIO_CLIENT.disconnect()
        await asyncio.to_thread(MEMORY.close)


if __name__ == "__main__":
    asyncio.run(run())
