#!/usr/bin/env python3
"""Fast, ground-truth memory evaluation.

Why this exists (and raw_full_trace.py is not enough): the soak harness grades
*structure* — right pathway, no "IDK", no "I can't", no action error — but never
compares the answer to what is actually true. So "Your car is at level 3" when
the user said "10 B" scores a clean pass. This harness instead encodes the
correct answer as per-turn assertions (must_contain / must_not_contain), replays
the *messy* phrasing real usage exposed, and includes corrections/supersessions.

It drives the same live TurnOrchestrator, but stays under a minute: ~a dozen
turns, short idle waits, and no 60s reminder-firing sleep. Ground truth is the
judge; the transcript is printed so a human stays in the loop for the fuzzy
quality that assertions can't capture.

Run (needs llama.cpp + Ollama; wipes only its own temp DB, never live memory):
    venv/bin/python memory_eval.py
Exit code is non-zero if any memory-category assertion fails.
"""

import asyncio
import os
import re
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")

import llama_client
from brain.action_runner import ActionRunner
from brain.config import load_brain_config
from brain.health import HealthReporter
from brain.intent_router import IntentRouter
from brain.memory_controller import MemoryController
from brain.orchestrator import TurnOrchestrator
from brain.planner import Planner
from generator import ResponseGenerator
from tools.websearch import DEFAULT_SEARCH_CLIENT

# Each case: what the user says, plus the ground truth the reply must honour.
#   must_contain      — every string must appear (space-insensitive, lowercased)
#   must_not_contain  — none may appear (catches hallucinated/stale values)
#   expect_pathway    — optional routing check
#   category          — "memory" gates the exit code; "scope" is a known
#                       out-of-memory-scope probe reported but not gating.
CASES = [
    # --- store from messy, real phrasing -------------------------------------
    {
        "prompt": "can you remember that I have a car parked in 10 B of the parking LOT",
        "category": "memory",
        "must_not_contain": ["level 3", "level 2", "floor 3", "spot 3"],
    },
    {
        "prompt": "where's my car?",
        "category": "memory",
        "must_contain": ["10 B"],
        "must_not_contain": ["level 3", "IDK", "don't know", "do not know", "not sure"],
    },
    # --- supersede: a new value must replace the old, not coexist ------------
    {
        "prompt": "actually I moved it — my car is now in 10 C",
        "category": "memory",
        "must_not_contain": ["level 3"],
    },
    {
        "prompt": "so where is my car parked now?",
        "category": "memory",
        "must_contain": ["10 C"],
        "must_not_contain": ["10 B", "level 3", "both", "conflicting"],
    },
    # --- identity + broad self-recall (the "IDK about myself" failure) -------
    {
        "prompt": "btw my name is Mira",
        "category": "memory",
        "must_not_contain": ["cannot save", "can't save", "unable to"],
    },
    {
        "prompt": "what do you actually know about me?",
        "category": "memory",
        "must_contain": ["Mira"],
        "must_not_contain": [
            "IDK",
            "don't know anything",
            "do not know anything",
            "nothing",
        ],
    },
    {
        "prompt": "what's my name again?",
        "category": "memory",
        "must_contain": ["Mira"],
        "must_not_contain": ["IDK", "don't know", "do not know"],
    },
    # --- no fabricated citations reach the user ------------------------------
    {
        "prompt": "remind me where my car is one more time",
        "category": "memory",
        "must_contain": ["10 C"],
        # a citation to a non-existent/echoed id would show as bracketed junk
        "must_not_contain": ["[memory:item:12]", "[item:", "level 3"],
    },
    # --- known out-of-memory-scope probes (reported, do not gate) -----------
    {
        "prompt": "remind me in one minute to stretch",
        "category": "scope",
        # Spelled-out durations ("one minute") now resolve. The forbidden set is
        # the refusal wording the parser used to emit — its absence confirms the
        # reminder was actually set rather than rejected.
        "must_not_contain": [
            "ACTION FAILED",
            "needs an exact time",
            "Nothing was stored",
            "cannot set",
            "can't set",
            "requires an exact time",
            "relative duration",
        ],
    },
]


def _norm(text: str) -> str:
    """Lowercase and strip whitespace so '10 B' matches '10b' / '10  B'."""
    return re.sub(r"\s+", "", (text or "").lower())


def _judge(response: str, case: dict) -> list[str]:
    """Return a list of assertion failures (empty == pass)."""
    failures = []
    norm = _norm(response)
    for needle in case.get("must_contain", []):
        if _norm(needle) not in norm:
            failures.append(f"missing required: {needle!r}")
    for needle in case.get("must_not_contain", []):
        if _norm(needle) in norm:
            failures.append(f"contains forbidden: {needle!r}")
    return failures


async def run_eval() -> int:
    # Pin all stores to a throwaway dir BEFORE building MemoryController so live
    # user memory is never touched.
    tmp = Path(tempfile.mkdtemp(prefix="amtavla-eval-"))
    os.environ["AMTAVLA_DB_DIR"] = str(tmp / "db")
    os.environ["AMTAVLA_CATALOG_DB"] = str(tmp / "db" / "memory_catalog.db")
    os.environ["AMTAVLA_VECTOR_DB"] = str(tmp / "db" / "ltm_vectors.db")
    (tmp / "db").mkdir(parents=True, exist_ok=True)

    config = load_brain_config()
    memory = MemoryController()
    search_client = DEFAULT_SEARCH_CLIENT
    action_runner = ActionRunner(search_client=search_client, memory_client=memory)
    orchestrator = TurnOrchestrator(
        router=IntentRouter(config),
        memory=memory,
        planner=Planner(max_steps=int(config.get("routing", {}).get("max_plan_steps", 5))),
        action_runner=action_runner,
        response_generator=ResponseGenerator(),
        health_reporter=HealthReporter(memory, search_client=search_client),
        config=config,
    )

    print("Starting llama-server...")
    await asyncio.to_thread(llama_client._ensure_server_running)
    await asyncio.to_thread(memory.clear_all_memory)
    print("Ready. Running memory eval.\n")

    results = []
    for index, case in enumerate(CASES, 1):
        turn = await orchestrator.process(
            case["prompt"], session_id="eval", input_source="script"
        )
        # Let the async commit land so the next turn can recall it, but cap the
        # wait so the whole run stays fast.
        await asyncio.to_thread(memory.wait_for_idle, 8.0)
        response = turn.response or ""
        failures = _judge(response, case)
        status = "PASS" if not failures else "FAIL"
        results.append((case, status, failures))

        print(f"[{index:02d}] ({case['category']}) > {case['prompt']}")
        print(f"     reply: {response.replace(chr(10), ' / ')[:200]}")
        print(f"     {status}" + (f" — {'; '.join(failures)}" if failures else ""))
        print()

    mem = [r for r in results if r[0]["category"] == "memory"]
    scope = [r for r in results if r[0]["category"] == "scope"]
    mem_fail = [r for r in mem if r[1] == "FAIL"]
    scope_fail = [r for r in scope if r[1] == "FAIL"]

    print("=" * 60)
    print(f"MEMORY:  {len(mem) - len(mem_fail)}/{len(mem)} passed")
    print(f"SCOPE:   {len(scope) - len(scope_fail)}/{len(scope)} passed (non-gating)")
    if mem_fail:
        print("\nMemory failures:")
        for case, _, failures in mem_fail:
            print(f"  - {case['prompt']!r}: {'; '.join(failures)}")
    return 1 if mem_fail else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(run_eval()))
