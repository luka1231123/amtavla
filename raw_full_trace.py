#!/usr/bin/env python3
"""Live conversation probe for the full turn loop.

Drives the real TurnOrchestrator through scenario suites grouped by capability
area, annotates each turn with likely capability gaps (wrong routing, missing
tool use, IDK on answerable prompts, "I can't" phrasing, failed actions), and
writes a ranked gap_report.md next to the usual raw trace logs. Requires
llama.cpp + Ollama running; there is no offline mode.
"""

import asyncio
import json
import os
import re
import sys
import time
import traceback
import urllib.request
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

# Scenario turn keys:
#   prompt          — what the user says ("/..." runs an operator command)
#   expect_pathway  — routed pathway must match exactly
#   expect_action   — this action type must appear in the executed plan
#   answerable      — an "IDK" reply counts as a gap (context was stored earlier)
#   expect_clarify  — the reply itself must be one clarifying question
SCENARIOS = [
    {
        "area": "memory_recall",
        "turns": [
            {"prompt": "hi there"},
            {"prompt": "Remember this: my name is Mira."},
            {"prompt": "Remember this: my bike is blue and parked in garage B2."},
            {"prompt": "What is my name?", "answerable": True},
            {"prompt": "Where is my bike parked?", "answerable": True},
            {"prompt": "What color is my bike?", "answerable": True},
        ],
    },
    {
        "area": "general_knowledge",
        "turns": [
            {"prompt": "What is 17 * 19?", "expect_action": "CALCULATE"},
            {"prompt": "Who wrote Pride and Prejudice?"},
            {"prompt": "Can you explain recursion simply?"},
        ],
    },
    {
        "area": "web_factual",
        "turns": [
            {
                "prompt": "What's the latest Python release?",
                "expect_action": "SEARCH",
            },
        ],
    },
    {
        "area": "summarize",
        "turns": [
            {"prompt": "Remember this: I need to renew my passport."},
            {"prompt": "Remember this: the flat viewing is in Vake on Saturday."},
            {
                "prompt": "Make a compact checklist from my notes.",
                "expect_pathway": "summarize_reply",
                "expect_action": "SUMMARIZE",
                "answerable": True,
            },
            {
                "prompt": "Summarize what you know about my week in 3 bullets.",
                "expect_action": "SUMMARIZE",
                "answerable": True,
            },
        ],
    },
    {
        "area": "reminders",
        "turns": [
            {
                "prompt": "Remind me to stretch in 1 minute.",
                "expect_pathway": "reminder_reply",
                "expect_action": "REMINDER",
            },
            {
                "prompt": "Remind me to call the dentist tomorrow morning.",
                "expect_pathway": "reminder_reply",
                "expect_action": "REMINDER",
            },
            {"prompt": "Remind me where my bike is.", "answerable": True},
        ],
    },
    {
        "area": "notes_files",
        "turns": [
            {
                "prompt": "List files in the current directory.",
                "expect_pathway": "notes_reply",
                "expect_action": "NOTE_READ",
            },
            {
                "prompt": "Read README.md and tell me what this project is.",
                "expect_action": "NOTE_READ",
            },
        ],
    },
    {
        "area": "clarify",
        "turns": [
            {"prompt": "Fix it.", "expect_clarify": True},
            {"prompt": "Can you make that thing better?", "expect_clarify": True},
        ],
    },
    {
        "area": "research",
        "turns": [
            {
                "prompt": "Do a deep dive on local-first sync engines for me.",
                "expect_pathway": "research_reply",
                "expect_action": "RESEARCH",
            },
        ],
    },
    {
        "area": "creativity",
        "turns": [
            {"prompt": "Brainstorm three name ideas for a note-taking app."},
        ],
    },
    {
        "area": "executive",
        "turns": [
            {"prompt": "I promised Nino I'll send the invoice by Friday."},
            {"prompt": "/idle"},
            {"prompt": "/brief"},
        ],
    },
]

_CANT_RE = re.compile(
    r"\b(i can'?t|i cannot|i'?m unable|i am unable|i don'?t have (?:the ability|access)|as an ai)\b",
    re.IGNORECASE,
)


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


def annotate_gaps(area: str, spec: dict, turn) -> list[dict]:
    """Heuristic per-turn gap detection; each gap is one dict for the report."""
    gaps = []
    prompt = spec["prompt"]
    response = turn.response or ""
    pathway = turn.route.pathway if turn.route else ""
    plan_actions = [action.action_type.value for action in turn.plan.actions]
    result_errors = [
        f"{result.action_type.value}: {result.error}"
        for result in turn.action_results
        if not result.ok
    ]

    def gap(kind: str, severity: str, detail: str):
        gaps.append(
            {
                "area": area,
                "kind": kind,
                "severity": severity,
                "prompt": prompt,
                "detail": detail,
                "pathway": pathway,
                "plan": plan_actions,
                "response": response[:200],
            }
        )

    if turn.status != "completed":
        gap("turn_failed", "high", f"status={turn.status} error={turn.error}")
    expected_pathway = spec.get("expect_pathway")
    if expected_pathway and pathway != expected_pathway:
        gap(
            "routing_miss",
            "high",
            f"expected pathway {expected_pathway}, got {pathway}",
        )
    expected_action = spec.get("expect_action")
    if expected_action and expected_action not in plan_actions:
        gap(
            "missing_tool_use",
            "high",
            f"expected {expected_action} in plan, got {plan_actions or 'no actions'}",
        )
    if spec.get("answerable") and response.strip().startswith("IDK"):
        gap("idk_on_answerable", "high", "answerable prompt got IDK")
    if spec.get("expect_clarify"):
        if "?" not in response or response.strip().startswith("IDK"):
            gap(
                "no_clarifying_question",
                "medium",
                "vague prompt did not yield a clarifying question",
            )
    if expected_action and plan_actions and set(plan_actions) == {"THINK"}:
        gap("bare_think_fallback", "medium", "plan degraded to THINK only")
    for warning in turn.plan.warnings:
        if "Unsupported" in warning:
            gap("unsupported_action", "medium", warning)
    if _CANT_RE.search(response):
        gap("claims_no_capability", "medium", "reply claims it cannot do the task")
    for error in result_errors:
        gap("action_error", "medium", error)
    return gaps


def write_gap_report(run_dir: Path, gaps: list[dict], turn_log: list[dict]):
    path = run_dir / "gap_report.md"
    counts: dict[str, int] = {}
    for item in gaps:
        counts[item["kind"]] = counts.get(item["kind"], 0) + 1
    lines = [
        "# Gap Report",
        "",
        f"Run: {run_dir.name} — {len(turn_log)} probed turns, "
        f"{len(gaps)} gaps flagged.",
        "",
        "## Gaps by kind",
        "",
    ]
    if counts:
        lines.append("| kind | count |")
        lines.append("| --- | --- |")
        for kind, count in sorted(counts.items(), key=lambda kv: -kv[1]):
            lines.append(f"| {kind} | {count} |")
    else:
        lines.append("No gaps flagged.")
    severity_order = {"high": 0, "medium": 1, "low": 2}
    lines += ["", "## Flagged turns", ""]
    for item in sorted(gaps, key=lambda g: severity_order.get(g["severity"], 3)):
        lines += [
            f"### [{item['severity']}] {item['kind']} — {item['area']}",
            "",
            f"- Prompt: {item['prompt']}",
            f"- Detail: {item['detail']}",
            f"- Pathway: {item['pathway']} | Plan: {item['plan']}",
            f"- Response: {item['response']}",
            "",
        ]
    lines += ["## All probed turns", ""]
    lines.append("| # | area | pathway | plan | gaps |")
    lines.append("| --- | --- | --- | --- | --- |")
    for entry in turn_log:
        lines.append(
            f"| {entry['index']} | {entry['area']} | {entry['pathway']} "
            f"| {', '.join(entry['plan']) or '-'} | {entry['gap_count']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def server_model_props() -> dict:
    """Read llama.cpp's live model metadata without relying on log output."""
    try:
        with urllib.request.urlopen("http://127.0.0.1:8085/props", timeout=5) as response:
            payload = json.loads(response.read())
            return payload if isinstance(payload, dict) else {"raw": payload}
    except Exception as exc:
        return {"error": str(exc)}


def validate_soak_run(run_dir: Path, config: dict, model_props: dict, gaps: list[dict], turn_log: list[dict]) -> dict:
    """Produce a machine-readable verdict for the real, completed soak run."""
    expected_model = config.get("llm", {}).get("model_filename", "")
    serialized_props = json.dumps(model_props).lower()
    model_confirmed = bool(expected_model) and expected_model.lower() in serialized_props
    incomplete_turns = [entry["index"] for entry in turn_log if entry["pathway"] == "?"]
    high_gaps = [gap for gap in gaps if gap["severity"] == "high"]
    verdict = {
        "passed": model_confirmed and not incomplete_turns and not high_gaps,
        "expected_model": expected_model,
        "model_confirmed_by_live_server": model_confirmed,
        "turns_completed": len(turn_log),
        "incomplete_turns": incomplete_turns,
        "high_severity_gap_count": len(high_gaps),
        "high_severity_gaps": high_gaps,
        "server_props": model_props,
    }
    (run_dir / "validation.json").write_text(
        json.dumps(verdict, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    return verdict


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


async def _run_command(memory, orchestrator, prompt: str, writer, turn_index: int):
    if prompt.startswith("/brain"):
        parts = prompt.split()
        mode = parts[1] if len(parts) > 1 else "status"
        return await asyncio.to_thread(memory.get_debug_info, mode)
    if prompt.startswith("/health"):
        health = await asyncio.to_thread(orchestrator.health_reporter.snapshot)
        return render_health(health)
    if prompt.startswith("/ask"):
        forced = await asyncio.to_thread(memory.force_proactive_ask)
        return forced.get("prompt") or "No pending insight is ready."
    if prompt.startswith("/idle"):
        result = await asyncio.to_thread(memory.run_idle_now)
        writer.block("idle", f"TURN {turn_index} MANUAL IDLE", result)
        return "Idle maintenance run complete.\n" + render(result)
    if prompt.startswith("/brief"):
        return await asyncio.to_thread(memory.daily_brief)
    if prompt.startswith("/delete"):
        await asyncio.to_thread(memory.clear_all_memory)
        return "All memory databases cleared."
    return f"Unknown command: {prompt}"


async def run_trace():
    run_dir = Path("logs") / "raw_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    # The probe clears memory by design. Pin every store to this run directory
    # before constructing MemoryController so live user memory is never touched.
    probe_db_dir = (run_dir / "db").resolve()
    probe_db_dir.mkdir(parents=True, exist_ok=True)
    os.environ["AMTAVLA_DB_DIR"] = str(probe_db_dir)
    os.environ["AMTAVLA_CATALOG_DB"] = str(probe_db_dir / "memory_catalog.db")
    os.environ["AMTAVLA_VECTOR_DB"] = str(probe_db_dir / "ltm_vectors.db")
    writer = TraceWriter(run_dir)
    print(f"Conversation probe started: {run_dir}")

    config = load_brain_config()
    memory = MemoryController()
    search_client = TracedSearchClient(writer)
    action_runner = ActionRunner(search_client=search_client, memory_client=memory)

    proactive_messages: list[str] = []

    def proactive_hook(message: str):
        proactive_messages.append(message)
        print(f"\n[PROACTIVE] {message}\n")
        writer.line("idle", f"PROACTIVE PUSH: {message}")

    memory.set_proactive_hook(proactive_hook)

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

    gaps: list[dict] = []
    turn_log: list[dict] = []
    short_reminder_set_at: float | None = None

    try:
        print("Starting llama-server...")
        await asyncio.to_thread(llama_client._ensure_server_running)
        model_props = await asyncio.to_thread(server_model_props)
        writer.block("brain", "LIVE SERVER MODEL PROPS", model_props)
        print("llama-server ready.")
        await asyncio.to_thread(memory.clear_all_memory)
        writer.line("brain", "Memory cleared at start")

        turn_index = 0
        for scenario in SCENARIOS:
            area = scenario["area"]
            print(f"\n=== SCENARIO: {area} ===")
            writer.line("session", f"=== SCENARIO: {area} ===")
            for spec in scenario["turns"]:
                turn_index += 1
                prompt = spec["prompt"]
                print(f"\n[{turn_index:02d}] > {prompt}")
                writer.line("session", f"TURN {turn_index} USER: {prompt}")

                if prompt.startswith("/"):
                    response = await _run_command(
                        memory, orchestrator, prompt, writer, turn_index
                    )
                    print(response)
                    writer.block("session", f"TURN {turn_index} ASSISTANT", response)
                    writer.block("brain", f"TURN {turn_index} COMMAND", response)
                    turn_log.append(
                        {
                            "index": turn_index,
                            "area": area,
                            "pathway": "(command)",
                            "plan": [],
                            "gap_count": 0,
                        }
                    )
                    continue

                started = time.perf_counter()
                turn = await orchestrator.process(
                    prompt,
                    session_id="probe",
                    input_source="script",
                )
                await asyncio.to_thread(memory.wait_for_idle, 30.0)
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                print(turn.response)
                _write_turn(writer, turn, turn_index, elapsed_ms)

                if "in 1 minute" in prompt.lower():
                    short_reminder_set_at = time.time()

                turn_gaps = annotate_gaps(area, spec, turn)
                gaps.extend(turn_gaps)
                for item in turn_gaps:
                    print(f"  [GAP:{item['severity']}] {item['kind']}: {item['detail']}")
                turn_log.append(
                    {
                        "index": turn_index,
                        "area": area,
                        "pathway": turn.route.pathway if turn.route else "?",
                        "plan": [a.action_type.value for a in turn.plan.actions],
                        "gap_count": len(turn_gaps),
                    }
                )

        # End-to-end reminder firing: wait out the 1-minute reminder, then let
        # the idle worker fire it through the proactive channel.
        if short_reminder_set_at is not None:
            wait_for = max(0.0, short_reminder_set_at + 65 - time.time())
            print(f"\nWaiting {wait_for:.0f}s for the 1-minute reminder to come due...")
            await asyncio.sleep(wait_for)
            await asyncio.to_thread(memory.run_idle_now)
            if not any(m.startswith("Reminder") for m in proactive_messages):
                gaps.append(
                    {
                        "area": "reminders",
                        "kind": "reminder_never_fired",
                        "severity": "high",
                        "prompt": "Remind me to stretch in 1 minute.",
                        "detail": "due reminder did not arrive via proactive push",
                        "pathway": "reminder_reply",
                        "plan": ["REMINDER"],
                        "response": "",
                    }
                )

        # Give queued research a chance to run and report back.
        await asyncio.to_thread(memory.run_idle_now)
        if not any(m.startswith("Research") for m in proactive_messages):
            gaps.append(
                {
                    "area": "research",
                    "kind": "research_never_reported",
                    "severity": "medium",
                    "prompt": "Do a deep dive on local-first sync engines for me.",
                    "detail": "queued research produced no proactive result message",
                    "pathway": "research_reply",
                    "plan": ["RESEARCH"],
                    "response": "",
                }
            )

        final_dump = await asyncio.to_thread(memory.get_brain_dump, "full", 200)
        writer.block("brain", "FINAL BRAIN DUMP", final_dump)
        writer.block("memory", "FINAL BRAIN DUMP", final_dump)

        report_path = write_gap_report(run_dir, gaps, turn_log)
        verdict = validate_soak_run(run_dir, config, model_props, gaps, turn_log)
        print(f"\nProbe complete: {len(turn_log)} turns, {len(gaps)} gaps flagged.")
        print(f"Gap report: {report_path}")
        print(
            "Validation: "
            f"{'PASS' if verdict['passed'] else 'FAIL'} "
            f"(live model confirmed={verdict['model_confirmed_by_live_server']}, "
            f"high gaps={verdict['high_severity_gap_count']})"
        )
        print(f"Logs saved in {run_dir}")
    except Exception as exc:
        print(f"FATAL ERROR: {exc}")
        writer.block("brain", "FATAL", traceback.format_exc())
        write_gap_report(run_dir, gaps, turn_log)
        raise
    finally:
        await asyncio.to_thread(memory.wait_for_idle, 3.0)
        await asyncio.to_thread(memory.close)
        writer.close()


def main():
    asyncio.run(run_trace())


if __name__ == "__main__":
    main()
