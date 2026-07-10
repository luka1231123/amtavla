#!/usr/bin/env python3

import asyncio
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, ".")

import llama_client
from brain.action_runner import ActionRunner
from brain.config import load_brain_config
from brain.health import HealthReporter, render_health
from brain.intent_router import IntentRouter
from brain.memory_controller import MemoryController
from brain.orchestrator import TurnOrchestrator
from brain.planner import Planner
from generator import ResponseGenerator
from tools.websearch import DEFAULT_SEARCH_CLIENT


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


def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def render(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, indent=2, ensure_ascii=True, default=str)


class TraceWriter:
    def __init__(self, run_dir: Path):
        self.files = {
            "brain": (run_dir / "brain_timeline.txt").open("w", encoding="utf-8"),
            "session": (run_dir / "session_transcript.txt").open(
                "w", encoding="utf-8"
            ),
            "memory": (run_dir / "memory_chronological.txt").open(
                "w", encoding="utf-8"
            ),
            "web": (run_dir / "web_searches.txt").open("w", encoding="utf-8"),
            "idle": (run_dir / "idle_processes.txt").open("w", encoding="utf-8"),
            "thoughts": (run_dir / "thoughts.txt").open("w", encoding="utf-8"),
        }

    def line(self, channel: str, message: str):
        file = self.files[channel]
        file.write(f"[{timestamp()}] {message}\n")
        file.flush()

    def block(self, channel: str, title: str, body):
        self.line(channel, title)
        for line in render(body).splitlines():
            self.line(channel, f"  {line}")

    def close(self):
        for file in self.files.values():
            file.close()


class TracedSearchClient:
    def __init__(self, writer: TraceWriter):
        self.writer = writer

    def search(self, query: str, *, cache=None):
        started = time.perf_counter()
        results = DEFAULT_SEARCH_CLIENT.search(query, cache=cache)
        self.writer.block(
            "web",
            f"SEARCH {query!r} ({int((time.perf_counter() - started) * 1000)} ms)",
            [item.to_dict() for item in results],
        )
        return results

    def health(self):
        return DEFAULT_SEARCH_CLIENT.health()


def _write_turn(writer: TraceWriter, turn, turn_index: int, elapsed_ms: int):
    writer.block(
        "session",
        f"TURN {turn_index} ASSISTANT",
        turn.response,
    )
    writer.block(
        "memory",
        f"TURN {turn_index} COMMIT PAYLOAD",
        turn.to_memory_trace(),
    )
    writer.block(
        "brain",
        f"TURN {turn_index} COMPLETE ({elapsed_ms} ms)",
        {
            "turn_id": turn.turn_id,
            "status": turn.status,
            "route": turn.route.to_dict() if turn.route else None,
            "plan": turn.plan.to_dict(),
            "actions": [item.to_dict() for item in turn.action_results],
            "response_source_ids": turn.response_source_ids,
            "error": turn.error,
        },
    )
    if turn.plan.thinking:
        writer.block(
            "thoughts",
            f"TURN {turn_index} PLANNER THINKING",
            turn.plan.thinking,
        )


async def run_trace():
    run_dir = Path("logs") / "raw_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = TraceWriter(run_dir)
    print(f"Raw full trace started: {run_dir}")

    config = load_brain_config()
    memory = MemoryController()
    search_client = TracedSearchClient(writer)
    action_runner = ActionRunner(search_client=search_client, memory_client=memory)

    def debug_hook(event_type: str, event: dict):
        writer.block("brain", f"TRACE {event_type}", event)

    orchestrator = TurnOrchestrator(
        router=IntentRouter(config),
        memory=memory,
        planner=Planner(
            max_steps=int(config.get("routing", {}).get("max_plan_steps", 5))
        ),
        action_runner=action_runner,
        response_generator=ResponseGenerator(),
        health_reporter=HealthReporter(memory, search_client=search_client),
        debug_hook=debug_hook,
        config=config,
    )

    try:
        print("Starting llama-server...")
        await asyncio.to_thread(llama_client._ensure_server_running)
        print("llama-server ready.")
        await asyncio.to_thread(memory.clear_all_memory)
        writer.line("brain", "Memory cleared at start")

        for turn_index, prompt in enumerate(PROMPTS, 1):
            print(f"\n[{turn_index:02d}] > {prompt}")
            writer.line("session", f"TURN {turn_index} USER: {prompt}")
            lowered = prompt.strip().lower()

            if lowered in {"exit", "quit", "q"}:
                print("Goodbye.")
                writer.line("session", "ASSISTANT: Goodbye.")
                break

            if prompt.startswith("/brain"):
                parts = prompt.split()
                mode = parts[1] if len(parts) > 1 else "status"
                response = await asyncio.to_thread(memory.get_debug_info, mode)
            elif prompt.startswith("/health"):
                health = await asyncio.to_thread(orchestrator.health_reporter.snapshot)
                response = render_health(health)
            elif prompt.startswith("/ask"):
                forced = await asyncio.to_thread(memory.force_proactive_ask)
                response = forced.get("prompt") or "No pending insight is ready."
            elif prompt.startswith("/idle"):
                result = await asyncio.to_thread(memory.run_idle_now)
                response = "Idle maintenance run complete.\n" + render(result)
                writer.block("idle", f"TURN {turn_index} MANUAL IDLE", result)
            elif prompt.startswith("/delete"):
                await asyncio.to_thread(memory.clear_all_memory)
                response = "All memory databases cleared."
            else:
                started = time.perf_counter()
                turn = await orchestrator.process(
                    prompt,
                    session_id="raw_full_trace",
                    input_source="script",
                )
                await asyncio.to_thread(memory.wait_for_idle, 30.0)
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                response = turn.response
                _write_turn(writer, turn, turn_index, elapsed_ms)

            print(response)
            if prompt.startswith("/"):
                writer.block("session", f"TURN {turn_index} ASSISTANT", response)
                writer.block("brain", f"TURN {turn_index} COMMAND", response)

            if turn_index % 4 == 0:
                idle_result = await asyncio.to_thread(memory.run_idle_now)
                writer.block(
                    "idle",
                    f"PERIODIC IDLE AFTER TURN {turn_index}",
                    idle_result,
                )

        final_dump = await asyncio.to_thread(memory.get_brain_dump, "full", 200)
        writer.block("brain", "FINAL BRAIN DUMP", final_dump)
        writer.block("memory", "FINAL BRAIN DUMP", final_dump)
        print(f"\nRaw trace complete. Logs saved in {run_dir}")
    except Exception as exc:
        print(f"FATAL ERROR: {exc}")
        writer.block("brain", "FATAL", traceback.format_exc())
        raise
    finally:
        await asyncio.to_thread(memory.wait_for_idle, 3.0)
        await asyncio.to_thread(memory.close)
        writer.close()


def main():
    asyncio.run(run_trace())


if __name__ == "__main__":
    main()
