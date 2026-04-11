import asyncio
import logging
import re
import sys

sys.path.insert(0, ".")

import llama_client
import socketio
from brain.config import load_brain_config
from brain.intent_router import IntentRouter
from brain.memory_controller import MemoryController
from brain.planner import generate_plan
from generator import generate_response
from logging_setup import configure_logging
from tools.websearch import tool_websearch

configure_logging()
logger = logging.getLogger("amtavla.main")
CONFIG = load_brain_config()
MEMORY = MemoryController()
ROUTER = IntentRouter(CONFIG)
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


def _clip_text(value, max_chars: int = 3000):
    if value is None:
        return ""
    text = str(value)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "... [truncated]"


def _build_debug_payload(event_type: str, payload: dict) -> dict:
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

    return {
        "type": event_type,
        "payload": {
            "summary": summary,
            "data": payload,
        },
    }


def _send_debug_event(event_type: str, payload: dict):
    _dispatch_socket_emit("debug_event", _build_debug_payload(event_type, payload))


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


def _is_short_ack_or_smalltalk(text: str) -> bool:
    tokens = re.findall(r"[a-z0-9']+", (text or "").lower())
    if len(tokens) <= 2:
        ack = {
            "hi",
            "hiya",
            "hey",
            "yo",
            "hello",
            "sup",
            "wazzup",
            "ok",
            "okay",
            "no",
            "yes",
            "thanks",
            "thx",
        }
        return all(t in ack for t in tokens) if tokens else True
    return False


def execute_plan_step(
    action: str,
    detail: str,
    user_input: str,
    memory_text: str,
    search_cache: dict | None = None,
) -> tuple[str, str, str]:
    del memory_text
    if action == "SEARCH":
        query = detail.strip() or user_input
        raw = tool_websearch(query, cache=search_cache)
        result = _compress_search_result(query, raw)
        return action, query, result
    if action == "THINK":
        return action, detail, ""
    return action, detail, ""


async def _run_plan_steps(
    plan: list[tuple[str, str]],
    user_input: str,
    context_text: str,
    search_cache: dict | None = None,
):
    tasks = [
        asyncio.to_thread(
            execute_plan_step,
            action,
            detail,
            user_input,
            context_text,
            search_cache,
        )
        for action, detail in plan
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    out = []
    for idx, result in enumerate(results):
        if isinstance(result, Exception):
            action, detail = plan[idx]
            out.append((action, detail, f"Error: {result}"))
            continue
        out.append(result)
    return out


async def _route_to_plan(
    user_input: str,
    route: dict,
    context_text: str,
    search_cache: dict | None = None,
):
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
        return (
            plan,
            await _run_plan_steps(plan, user_input, context_text, search_cache),
            thinking,
        )

    plan, thinking = await asyncio.to_thread(
        generate_plan,
        user_input,
        context_text,
        route.get("intent"),
        pathway,
    )
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
    return (
        filtered,
        await _run_plan_steps(filtered, user_input, context_text, search_cache),
        thinking,
    )


def _has_any_context(context: dict) -> bool:
    if context.get("semantic_facts"):
        return True
    if context.get("episodic_context"):
        return True
    if context.get("ltm_context"):
        return True
    if (context.get("web_context") or "").strip():
        return True
    return False


def _build_todo_from_plan(plan: list[tuple[str, str]]) -> list[dict]:
    todos = []
    for idx, (action, detail) in enumerate(plan, 1):
        todos.append(
            {
                "task_id": f"task-{idx}",
                "action": action,
                "detail": detail,
                "priority": "medium",
            }
        )
    return todos


async def _connect_realtime(command_queue: asyncio.Queue[str]):
    global SIO_CLIENT
    sio = socketio.AsyncClient(reconnection=True, logger=False, engineio_logger=False)

    @sio.event
    async def connect():
        logger.info("Connected to phone server via Socket.IO")

    @sio.event
    async def disconnect():
        logger.warning("Disconnected from phone server")

    @sio.on("command_submitted")
    async def on_command_submitted(data):
        text = ((data or {}).get("text") or "").strip()
        if text:
            await command_queue.put(text)

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
    command_queue: asyncio.Queue[str] = asyncio.Queue()

    MEMORY.set_debug_hook(_send_debug_event)

    try:
        await asyncio.to_thread(llama_client._ensure_server_running)
    except Exception as e:
        print(f"[FATAL] Failed to start llama-server: {e}")
        print(
            "Please install/build llama.cpp server and ensure a .gguf model exists in ~/llama.cpp/models."
        )
        return

    await _connect_realtime(command_queue)

    stdin_reader_installed = False
    if sys.platform != "win32":

        def _on_stdin_ready():
            line = sys.stdin.readline()
            if line:
                command_queue.put_nowait(line.strip())

        ASYNC_LOOP.add_reader(sys.stdin, _on_stdin_ready)
        stdin_reader_installed = True

    print(
        "amtavla - CLI assistant (type 'exit' to quit, '/brain <mode>' debug, '/ask' proactive debug, '/idle' force idle, '/delete')\n"
    )
    print("Or use phone UI at http://127.0.0.1:8081\n")

    try:
        while True:
            user_input = await command_queue.get()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye.")
                break

            if user_input.startswith("/brain"):
                parts = user_input.split()
                mode = parts[1] if len(parts) > 1 else "status"
                debug_info = await asyncio.to_thread(MEMORY.get_debug_info, mode)
                print(debug_info)
                print()
                continue

            if user_input.startswith("/ask"):
                forced = await asyncio.to_thread(MEMORY.force_proactive_ask)
                prompt = (
                    forced.get("prompt")
                    or "No pending insight is ready for proactive ask."
                )
                if forced.get("insight_id"):
                    prompt += f" [insight_id={forced['insight_id']}]"
                print(f"{prompt}\n")
                continue

            if user_input.startswith("/idle"):
                result = await asyncio.to_thread(MEMORY.run_idle_now)
                print("Idle maintenance run complete.")
                print(f"Status: {result.get('status')}")
                print(f"Metrics: {result.get('metrics')}\n")
                continue

            if user_input.startswith("/delete"):
                await asyncio.to_thread(MEMORY.clear_all_memory)
                print("All memory databases cleared (episodic, semantic, insight).\n")
                continue

            try:
                await asyncio.to_thread(MEMORY.begin_foreground_turn)
                await asyncio.to_thread(MEMORY.note_user_activity)
                _send_debug_event(
                    "user_prompt", {"prompt": _clip_text(user_input, 2000)}
                )

                route = await asyncio.to_thread(ROUTER.route, user_input)
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
                    dump = await asyncio.to_thread(MEMORY.get_brain_dump, mode)
                    print(f"{dump}\n")
                    _send_response_to_ui(dump)
                    MEMORY.process_turn_async(
                        user_input,
                        dump,
                        trace={
                            "intent": route.get("intent", ""),
                            "pathway": route.get("pathway", ""),
                            "todo": [],
                            "context": {"brain_dump_mode": mode},
                            "session_id": "cli",
                        },
                    )
                    continue

                include_web = route.get("pathway") not in {
                    "remember_reply",
                    "memory_recall_reply",
                    "brain_dump_reply",
                    "direct_reply",
                }
                if route.get("intent") in {"smalltalk", "greeting"}:
                    include_web = False
                if _is_short_ack_or_smalltalk(user_input):
                    include_web = False
                turn_search_cache = {}
                context = await asyncio.to_thread(
                    MEMORY.get_context_for_prompt,
                    user_input,
                    include_web,
                    route.get("intent", ""),
                    route.get("pathway", ""),
                    turn_search_cache,
                )
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

                plan, plan_results, thinking = await _route_to_plan(
                    user_input,
                    route,
                    context_text,
                    turn_search_cache,
                )
                todo_list = _build_todo_from_plan(plan)
                _send_debug_event(
                    "plan",
                    {
                        "thinking": _clip_text(thinking, 800),
                        "steps": [
                            {"action": action, "detail": detail}
                            for action, detail in plan
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

                if route.get("intent") == "memory_recall" and not _has_any_context(
                    context
                ):
                    response = "IDK"
                else:
                    response = await asyncio.to_thread(
                        generate_response,
                        user_input,
                        plan,
                        plan_results,
                        context,
                        route.get("intent"),
                        route.get("pathway"),
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
                        "pathway": route.get("pathway", ""),
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
            finally:
                await asyncio.to_thread(MEMORY.end_foreground_turn)
    except (KeyboardInterrupt, EOFError):
        print("\nExiting.")
    finally:
        if stdin_reader_installed and ASYNC_LOOP is not None:
            ASYNC_LOOP.remove_reader(sys.stdin)
        if SIO_CLIENT is not None and SIO_CLIENT.connected:
            await SIO_CLIENT.disconnect()


if __name__ == "__main__":
    asyncio.run(run())
