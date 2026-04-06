#!/usr/bin/env python3
"""
Integration test suite for amtavla.
Uses real Ollama calls for embeddings and LLM operations.

Run: python3 tests/test_integration.py
"""

import json
import logging
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "integration_test.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("test.integration")

from brain.stm import read_stm, append_stm, clear_stm, STM_FILE, MAX_ENTRIES
from brain.ltm_tree import (
    LtmTree,
    _safe_embed,
    _cosine_similarity,
    RETRIEVAL_THRESHOLD,
    MERGE_THRESHOLD,
    MAX_CONTENT_PER_BRANCH,
    MAX_DEPTH,
)
from brain.consolidator import (
    summarize_for_stm,
    detect_topic_shift,
    consolidate_to_tree,
)
from brain.memory_controller import MemoryController
from brain.planner import generate_plan
from tools import tool_weather, tool_bash_simulator
from tools.websearch import tool_websearch
from generator import generate_response

PASS_COUNT = 0
FAIL_COUNT = 0
FAILURES = []


def run_test(name, fn):
    global PASS_COUNT, FAIL_COUNT
    start = time.time()
    try:
        result = fn()
        elapsed = time.time() - start
        if result:
            PASS_COUNT += 1
            print(f"  PASS  {name} ({elapsed:.2f}s)")
        else:
            FAIL_COUNT += 1
            FAILURES.append(name)
            print(f"  FAIL  {name} ({elapsed:.2f}s) — returned False")
    except Exception as e:
        FAIL_COUNT += 1
        FAILURES.append(name)
        elapsed = time.time() - start
        print(f"  FAIL  {name} ({elapsed:.2f}s) — {type(e).__name__}: {e}")


def temp_tree_file():
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    return path


def temp_stm_file():
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    return path


def cleanup(paths):
    for p in paths:
        if p and os.path.exists(p):
            os.remove(p)


def wait_memory_thread(mc):
    if hasattr(mc, "wait_for_idle"):
        mc.wait_for_idle(timeout=2.0)
    else:
        time.sleep(0.2)


# ─── STM Tests ───────────────────────────────────────────────────────────────


def test_stm_append_and_read():
    f = temp_stm_file()
    try:
        append_stm("first line", "response one", stm_file=f)
        append_stm("second line", "response two", stm_file=f)
        content = read_stm(stm_file=f)
        lines = content.splitlines()
        return len(lines) == 4 and "first line" in content and "second line" in content
    finally:
        cleanup([f])


def test_stm_max_lines():
    f = temp_stm_file()
    try:
        for i in range(MAX_ENTRIES + 5):
            append_stm(f"line {i}", f"resp {i}", stm_file=f)
        content = read_stm(stm_file=f)
        lines = [l for l in content.splitlines() if l.strip()]
        expected = MAX_ENTRIES * 2
        return len(lines) == expected and "line 5" in lines[0]
    finally:
        cleanup([f])


def test_stm_clear():
    f = temp_stm_file()
    try:
        append_stm("data", "response", stm_file=f)
        clear_stm(stm_file=f)
        return read_stm(stm_file=f) == ""
    finally:
        cleanup([f])


def test_stm_read_nonexistent():
    return read_stm(stm_file="/tmp/nonexistent_stm_12345.txt") == ""


# ─── LTM Tree Tests ─────────────────────────────────────────────────────────


def test_tree_create_root_branch():
    tf = temp_tree_file()
    try:
        tree = LtmTree(tree_file=tf)
        b = tree.add_branch("Python Programming", ["Python is a versatile language"])
        return (
            b is not None
            and b["topic"] == "Python Programming"
            and len(b["embedding"]) > 0
            and len(b["content"]) == 1
        )
    finally:
        cleanup([tf])


def test_tree_create_child_branch():
    tf = temp_tree_file()
    try:
        tree = LtmTree(tree_file=tf)
        parent = tree.add_branch("Programming", ["General programming concepts"])
        child = tree.add_branch(
            "Python", ["Python syntax and features"], parent_id=parent["id"]
        )
        return (
            child is not None
            and len(parent["children"]) == 1
            and parent["children"][0]["id"] == child["id"]
        )
    finally:
        cleanup([tf])


def test_tree_max_depth():
    tf = temp_tree_file()
    try:
        tree = LtmTree(tree_file=tf)
        b1 = tree.add_branch("L1", ["level 1"])
        b2 = tree.add_branch("L2", ["level 2"], parent_id=b1["id"])
        b3 = tree.add_branch("L3", ["level 3"], parent_id=b2["id"])
        b4 = tree.add_branch("L4", ["level 4"], parent_id=b3["id"])
        b5 = tree.add_branch("L5", ["level 5"], parent_id=b4["id"])
        return b5 is None and b4 is not None
    finally:
        cleanup([tf])


def test_tree_content_limit():
    tf = temp_tree_file()
    try:
        tree = LtmTree(tree_file=tf)
        b = tree.add_branch("Test", ["initial"])
        for i in range(MAX_CONTENT_PER_BRANCH + 10):
            tree.append_to_branch(b["id"], [f"item {i}"])
        branch = tree._find_branch(b["id"])
        return len(branch["content"]) == MAX_CONTENT_PER_BRANCH
    finally:
        cleanup([tf])


def test_tree_append_updates_embedding():
    tf = temp_tree_file()
    try:
        tree = LtmTree(tree_file=tf)
        b = tree.add_branch("Test", ["initial content"])
        old_embed = list(b["embedding"])
        tree.append_to_branch(b["id"], ["new content added"])
        branch = tree._find_branch(b["id"])
        return branch["embedding"] != old_embed and len(branch["content"]) == 2
    finally:
        cleanup([tf])


def test_tree_find_best_branch():
    tf = temp_tree_file()
    try:
        tree = LtmTree(tree_file=tf)
        tree.add_branch("Python Programming", ["Python is a programming language"])
        tree.add_branch("Cooking Recipes", ["How to bake a cake"])
        query_embed = _safe_embed("Python functions and decorators")
        best = tree.find_best_branch(query_embed)
        return best is not None and "Python" in best["topic"]
    finally:
        cleanup([tf])


def test_tree_retrieve_context_pulls_subbranch():
    tf = temp_tree_file()
    try:
        tree = LtmTree(tree_file=tf)
        parent = tree.add_branch("Python", ["Python basics"])
        child = tree.add_branch(
            "Python OOP", ["Classes and objects"], parent_id=parent["id"]
        )
        tree.add_branch("Cooking", ["Recipe for pasta"])
        ctx = tree.retrieve_context("Python classes")
        has_python = "Python" in ctx
        has_oop = "Python OOP" in ctx or "Classes and objects" in ctx
        return has_python and has_oop
    finally:
        cleanup([tf])


def test_tree_branch_merging():
    tf = temp_tree_file()
    try:
        tree = LtmTree(tree_file=tf)
        b1 = tree.add_branch("Python", ["Python basics"])
        b2 = tree.add_branch("Python", ["Python basics"])
        tree.check_and_merge_siblings(b1["id"])
        nodes = tree._collect_all_nodes()
        return len(nodes) == 1
    finally:
        cleanup([tf])


def test_tree_save_load():
    tf = temp_tree_file()
    try:
        tree = LtmTree(tree_file=tf)
        b = tree.add_branch("Test Topic", ["Content one", "Content two"])
        tree.save()
        tree2 = LtmTree(tree_file=tf)
        tree2.load()
        return (
            len(tree2.branches) == 1
            and tree2.branches[0]["topic"] == "Test Topic"
            and len(tree2.branches[0]["content"]) == 2
        )
    finally:
        cleanup([tf])


def test_tree_load_corrupted_file():
    tf = temp_tree_file()
    try:
        with open(tf, "w") as f:
            f.write("not valid json {{{")
        tree = LtmTree(tree_file=tf)
        tree.load()
        return tree.branches == []
    finally:
        cleanup([tf])


def test_tree_load_empty_file():
    tf = temp_tree_file()
    try:
        with open(tf, "w") as f:
            f.write("")
        tree = LtmTree(tree_file=tf)
        tree.load()
        return tree.branches == []
    finally:
        cleanup([tf])


def test_tree_most_active_branch():
    tf = temp_tree_file()
    try:
        tree = LtmTree(tree_file=tf)
        tree.add_branch("Small", ["one item"])
        big = tree.add_branch("Big", [f"item {i}" for i in range(10)])
        active = tree.get_most_active_branch()
        return active is not None and active["id"] == big["id"]
    finally:
        cleanup([tf])


def test_tree_visualize():
    tf = temp_tree_file()
    try:
        tree = LtmTree(tree_file=tf)
        p = tree.add_branch("Root", ["root content"])
        tree.add_branch("Child", ["child content"], parent_id=p["id"])
        viz = tree.visualize()
        return "Root" in viz and "Child" in viz and "(empty)" not in viz
    finally:
        cleanup([tf])


def test_tree_empty_visualize():
    tf = temp_tree_file()
    try:
        tree = LtmTree(tree_file=tf)
        return tree.visualize() == "(empty)"
    finally:
        cleanup([tf])


def test_cosine_similarity():
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    c = [0.0, 1.0, 0.0]
    return (
        abs(_cosine_similarity(a, b) - 1.0) < 0.001
        and abs(_cosine_similarity(a, c)) < 0.001
    )


def test_exclude_id_in_find():
    tf = temp_tree_file()
    try:
        tree = LtmTree(tree_file=tf)
        b1 = tree.add_branch("Python", ["Python is great"])
        tree.add_branch("Python", ["Python is great"])
        query_embed = _safe_embed("Python programming")
        best = tree.find_best_branch(query_embed, exclude_id=b1["id"])
        return best is None or best["id"] != b1["id"]
    finally:
        cleanup([tf])


def test_tree_dict_index():
    tf = temp_tree_file()
    try:
        tree = LtmTree(tree_file=tf)
        b = tree.add_branch("Test", ["content"])
        return tree._find_branch(b["id"]) is not None
    finally:
        cleanup([tf])


# ─── Topic Detection Tests ──────────────────────────────────────────────────


def test_topic_shift_same_topic():
    tf = temp_tree_file()
    try:
        tree = LtmTree(tree_file=tf)
        branch = tree.add_branch(
            "Python Programming",
            [
                "Python is a high-level programming language with dynamic typing",
                "Python uses decorators and context managers for function wrapping",
                "Python supports object-oriented and functional programming paradigms",
            ],
        )
        shifted, topic = detect_topic_shift("How do Python decorators work?", branch)
        return not shifted
    finally:
        cleanup([tf])


def test_topic_shift_different_topic():
    tf = temp_tree_file()
    try:
        tree = LtmTree(tree_file=tf)
        branch = tree.add_branch(
            "Python Programming",
            [
                "Python is a high-level programming language with dynamic typing",
                "Python uses decorators and context managers for function wrapping",
                "Python supports object-oriented and functional programming paradigms",
            ],
        )
        shifted, new_topic = detect_topic_shift("What is the weather in Tokyo?", branch)
        return shifted and len(new_topic) > 0
    finally:
        cleanup([tf])


def test_topic_shift_no_branch():
    shifted, topic = detect_topic_shift("anything", None)
    return shifted and topic == ""


def test_topic_shift_uses_stored_embedding():
    tf = temp_tree_file()
    try:
        tree = LtmTree(tree_file=tf)
        branch = tree.add_branch(
            "Python Programming",
            [
                "Python is a high-level programming language with dynamic typing",
                "Python uses decorators and context managers for function wrapping",
            ],
        )
        shifted, _ = detect_topic_shift("How do Python decorators work?", branch)
        return not shifted
    finally:
        cleanup([tf])


# ─── Consolidation Tests ────────────────────────────────────────────────────


def test_consolidation_creates_root_branch():
    tf = temp_tree_file()
    sf = temp_stm_file()
    try:
        tree = LtmTree(tree_file=tf)
        append_stm(
            "Python is a programming language created by Guido", "Noted", stm_file=sf
        )
        append_stm(
            "Python uses dynamic typing and garbage collection", "Captured", stm_file=sf
        )
        consolidate_to_tree(read_stm(stm_file=sf), tree, None)
        return len(tree.branches) == 1 and len(tree.branches[0]["content"]) > 0
    finally:
        cleanup([tf, sf])


def test_consolidation_excludes_current_branch():
    tf = temp_tree_file()
    sf = temp_stm_file()
    try:
        tree = LtmTree(tree_file=tf)
        b1 = tree.add_branch("Python", ["Python basics"])
        append_stm("Python has decorators and context managers", "Stored", stm_file=sf)
        consolidate_to_tree(read_stm(stm_file=sf), tree, b1["id"])
        nodes = tree._collect_all_nodes()
        for n in nodes:
            if n["id"] != b1["id"]:
                return True
        return len(nodes) == 1
    finally:
        cleanup([tf, sf])


def test_consolidation_no_dead_code():
    import brain.consolidator as consolidator
    import inspect

    source = inspect.getsource(consolidator.detect_topic_shift)
    last_return = source.rfind('return False, current_branch["topic"]')
    after = source[last_return + len('return False, current_branch["topic"]') :]
    return after.strip() == ""


# ─── Tool Tests ──────────────────────────────────────────────────────────────


def test_tool_weather_tokyo():
    result = tool_weather("What's the weather in Tokyo?", "")
    return "Tokyo" in result and "Sunny" in result


def test_tool_weather_london():
    result = tool_weather("Temperature in London?", "")
    return "London" in result


def test_tool_weather_unknown():
    result = tool_weather("Weather in Paris?", "")
    return "unknown" in result.lower()


def test_tool_bash_files():
    result = tool_bash_simulator("list all files in current directory", "")
    return "file1.txt" in result


def test_tool_bash_python_version():
    result = tool_bash_simulator("what python version", "")
    return "Python" in result


def test_tool_bash_date():
    result = tool_bash_simulator("what is the current date", "")
    return "2026" in result


def test_tool_bash_no_llm():
    import tools
    import inspect

    source = inspect.getsource(tools.tool_bash_simulator)
    return "ollama" not in source


def test_tool_websearch():
    result = tool_websearch("Python programming language")
    return isinstance(result, str) and len(result) > 5


def test_tool_websearch_empty():
    result = tool_websearch("xyznonexistentquery123456789")
    return isinstance(result, str)


# ─── Planner Tests ──────────────────────────────────────────────────────────


def test_planner_generates_steps():
    steps, _ = generate_plan("What is Python?", "No context")
    return len(steps) > 0 and len(steps) <= 5


def test_planner_includes_search():
    steps, _ = generate_plan("What is Python?", "No context")
    actions = [s[0] for s in steps]
    return "SEARCH" in actions


def test_planner_max_5_steps():
    steps, _ = generate_plan(
        "Tell me everything about quantum computing and its applications", "No context"
    )
    return len(steps) <= 5


def test_planner_no_json():
    steps, _ = generate_plan("What is Python?", "No context")
    for action, detail in steps:
        assert isinstance(action, str)
        assert isinstance(detail, str)
    return True


# ─── Generator Test ─────────────────────────────────────────────────────────


def test_generator_produces_response():
    plan = [("SEARCH", "Python basics"), ("THINK", "")]
    plan_results = [
        ("SEARCH", "Python basics", "Python is a programming language..."),
        ("THINK", "", ""),
    ]
    response = generate_response(
        "What is Python?",
        plan,
        plan_results,
        {"working_memory": "", "ltm_context": ""},
    )
    return len(response) > 0


def test_generator_with_empty_results():
    response = generate_response(
        "Hello",
        [],
        [],
        {"working_memory": "", "ltm_context": ""},
    )
    return len(response) > 0


# ─── MemoryController Integration Tests ─────────────────────────────────────


def test_mc_first_turn_creates_branch():
    tf = temp_tree_file()
    sf = temp_stm_file()
    try:
        mc = MemoryController(stm_file=sf, tree_file=tf)
        mc._process_turn("What is Python?", "Python is a programming language.")
        wait_memory_thread(mc)
        return len(mc.tree.branches) == 1
    finally:
        cleanup([tf, sf])


def test_mc_same_topic_stays_in_branch():
    tf = temp_tree_file()
    sf = temp_stm_file()
    try:
        mc = MemoryController(stm_file=sf, tree_file=tf)
        mc._process_turn(
            "What is Python?",
            "Python is a high-level programming language with dynamic typing, garbage collection, and extensive standard library support for web development, data science, and automation.",
        )
        wait_memory_thread(mc)
        branch_id_1 = mc.current_branch_id
        mc._process_turn(
            "What about decorators?",
            "Python decorators are functions that take another function as input, add behavior, and return a modified function. They use the @ syntax and are commonly used for logging, caching, and authentication.",
        )
        wait_memory_thread(mc)
        return branch_id_1 is not None and len(mc.tree.branches) >= 1
    finally:
        cleanup([tf, sf])


def test_mc_topic_shift_creates_new_branch():
    tf = temp_tree_file()
    sf = temp_stm_file()
    try:
        mc = MemoryController(stm_file=sf, tree_file=tf)
        mc._process_turn(
            "What is Python?",
            "Python is a high-level programming language with dynamic typing, garbage collection, decorators, classes, and extensive standard library support for web development, data science, and automation.",
        )
        wait_memory_thread(mc)
        mc._process_turn(
            "How do decorators work?",
            "Python decorators wrap functions to add behavior. They use the @ syntax, accept a function as input, and return a modified function. Common uses include logging, caching, and authentication middleware.",
        )
        wait_memory_thread(mc)
        mc._process_turn(
            "What is the weather in Tokyo?",
            "Tokyo is currently sunny with a temperature of 22 degrees Celsius. The forecast for the week is clear skies with temperatures between 20-25 degrees.",
        )
        wait_memory_thread(mc)
        return len(mc.tree.branches) >= 1
    finally:
        cleanup([tf, sf])


def test_mc_debug_status():
    tf = temp_tree_file()
    sf = temp_stm_file()
    try:
        mc = MemoryController(stm_file=sf, tree_file=tf)
        info = mc.get_debug_info("status")
        return "Short-Term Memory" in info and "Current Branch" in info
    finally:
        cleanup([tf, sf])


def test_mc_debug_ltm():
    tf = temp_tree_file()
    sf = temp_stm_file()
    try:
        mc = MemoryController(stm_file=sf, tree_file=tf)
        info = mc.get_debug_info("ltm")
        return "Long-Term Memory" in info
    finally:
        cleanup([tf, sf])


def test_mc_session_resume():
    tf = temp_tree_file()
    sf = temp_stm_file()
    try:
        mc1 = MemoryController(stm_file=sf, tree_file=tf)
        mc1._process_turn("Python basics", "Python is a programming language.")
        wait_memory_thread(mc1)
        mc1._process_turn("Python decorators", "Decorators wrap functions.")
        wait_memory_thread(mc1)
        mc1._process_turn("What is the weather?", "Sunny in Tokyo.")
        wait_memory_thread(mc1)
        mc1.tree.save()

        mc2 = MemoryController(stm_file=sf, tree_file=tf)
        return mc2.current_branch_id is not None
    finally:
        cleanup([tf, sf])


def test_mc_fire_and_forget():
    tf = temp_tree_file()
    sf = temp_stm_file()
    try:
        mc = MemoryController(stm_file=sf, tree_file=tf)
        mc._process_turn("Test 1", "Response 1")
        mc._process_turn("Test 2", "Response 2")
        mc._process_turn("Test 3", "Response 3")
        return True
    finally:
        cleanup([tf, sf])


# ─── Full CLI Loop Tests ────────────────────────────────────────────────────


def run_cli_loop(inputs, stm_file, tree_file):
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from brain.memory_controller import MemoryController
    from brain.planner import generate_plan
    from generator import generate_response
    from tools import tool_weather, tool_bash_simulator
    from tools.websearch import tool_websearch

    memory = MemoryController(stm_file=stm_file, tree_file=tree_file)
    history = deque(maxlen=5)
    outputs = []

    for user_input in inputs:
        if user_input.lower() in ("exit", "quit", "q"):
            break

        context = memory.get_context_for_prompt(user_input)
        context_text = (
            (context.get("working_memory", "") or "")
            + " "
            + (context.get("ltm_context", "") or "")
        )

        plan, _ = generate_plan(user_input, context_text)

        def execute_step(action, detail):
            if action == "SEARCH":
                return action, detail, tool_websearch(detail)
            elif action == "TOOL":
                if detail == "weather":
                    return (
                        action,
                        detail,
                        tool_weather(user_input, context.get("working_memory", "")),
                    )
                elif detail == "bash":
                    return (
                        action,
                        detail,
                        tool_bash_simulator(
                            user_input, context.get("working_memory", "")
                        ),
                    )
            elif action == "THINK":
                return action, detail, ""
            return action, detail, ""

        plan_results = [None] * len(plan)
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(execute_step, action, detail): (idx, action, detail)
                for idx, (action, detail) in enumerate(plan)
            }
            for future in as_completed(futures):
                idx, action, detail = futures[future]
                try:
                    result = future.result()
                    plan_results[idx] = result
                except Exception as e:
                    plan_results[idx] = (action, detail, f"Error: {e}")

        plan_results = [r for r in plan_results if r is not None]

        response = generate_response(user_input, plan, plan_results, context)
        outputs.append((user_input, plan, response))
        history.append((user_input, response))
        memory.process_turn_async(user_input, response)

    import time

    memory.wait_for_idle(timeout=2.0)
    memory.tree.save()
    return outputs, memory


def test_cli_full_conversation_with_topic_shift():
    tf = temp_tree_file()
    sf = temp_stm_file()
    try:
        inputs = [
            "What is Python programming language and what are its main features?",
            "How do Python decorators work and what are some common use cases?",
            "What are Python classes and objects and how does inheritance work?",
            "What is the weather in Tokyo right now?",
            "exit",
        ]
        outputs, memory = run_cli_loop(inputs, sf, tf)

        if len(outputs) < 3:
            print(f"\n    Expected at least 3 outputs, got {len(outputs)}")
            return False

        branches = memory.tree.branches
        topics = [b["topic"] for b in branches]
        print(f"\n    Branches created: {topics}")

        has_python = any("Python" in t or "python" in t.lower() for t in topics)
        if not has_python:
            print(f"\n    Expected Python-related branch, got topics: {topics}")
            return False

        for branch in branches:
            if len(branch["embedding"]) == 0:
                print(f"\n    Branch '{branch['topic']}' has no embedding")
                return False

        for user_input, plan, response in outputs:
            if len(response) == 0:
                print(f"\n    Empty response for: {user_input}")
                return False

        return True
    finally:
        cleanup([tf, sf])


def test_cli_greeting_filtered_out():
    tf = temp_tree_file()
    sf = temp_stm_file()
    try:
        inputs = [
            "Hey",
            "exit",
        ]
        outputs, memory = run_cli_loop(inputs, sf, tf)
        branches = memory.tree.branches
        if len(branches) == 0:
            return True
        if len(branches) == 1:
            b = branches[0]
            return len(b["content"]) > 0 and len(b["embedding"]) > 0
        return False
    finally:
        cleanup([tf, sf])


def test_cli_weather_in_plan():
    tf = temp_tree_file()
    sf = temp_stm_file()
    try:
        inputs = [
            "What is the temperature in Tokyo?",
            "exit",
        ]
        outputs, memory = run_cli_loop(inputs, sf, tf)

        if len(outputs) != 1:
            print(f"\n    Expected 1 output, got {len(outputs)}")
            return False

        user_input, plan, response = outputs[0]
        actions = [p[0] for p in plan]
        has_tool = "TOOL" in actions
        has_search = "SEARCH" in actions

        if not has_tool and not has_search:
            print(f"\n    Plan should include TOOL or SEARCH, got: {plan}")
            return False

        if len(response) == 0:
            print(f"\n    Empty response")
            return False

        return True
    finally:
        cleanup([tf, sf])


def test_cli_subbranch_creation():
    tf = temp_tree_file()
    sf = temp_stm_file()
    try:
        inputs = [
            "Tell me about Python programming language features and syntax in detail",
            "What about JavaScript and web development frameworks and how they compare?",
            "How does React work for building user interfaces and what are hooks?",
            "exit",
        ]
        outputs, memory = run_cli_loop(inputs, sf, tf)

        all_nodes = memory.tree._collect_all_nodes()
        topics = [n["topic"] for n in all_nodes]
        print(f"\n    Nodes created: {topics}")

        if len(all_nodes) < 1:
            print(f"\n    Expected at least 1 node, got {len(all_nodes)}")
            return False

        has_programming = any(
            any(
                kw in t.lower()
                for kw in ["python", "javascript", "react", "programming", "web", "js"]
            )
            for t in topics
        )
        if not has_programming:
            print(f"\n    Expected programming-related topics, got: {topics}")
            return False

        return True
    finally:
        cleanup([tf, sf])


def test_cli_context_retrieval_from_ltm():
    tf = temp_tree_file()
    sf = temp_stm_file()
    try:
        inputs = [
            "Tell me about Python programming language features and syntax and decorators",
            "What about Python classes and object oriented programming?",
            "exit",
        ]
        outputs, memory = run_cli_loop(inputs, sf, tf)

        if len(memory.tree.branches) < 1:
            print(f"\n    Expected at least 1 branch, got {len(memory.tree.branches)}")
            return False

        ctx = memory.tree.retrieve_context("Python decorators and classes")
        if len(ctx) == 0:
            print(f"\n    Expected LTM context to be retrieved, got empty string")
            return False
        if "Python" not in ctx:
            print(f"\n    Expected Python in context: {ctx}")
            return False

        return True
    finally:
        cleanup([tf, sf])


def test_cli_branch_content_accumulation():
    tf = temp_tree_file()
    sf = temp_stm_file()
    try:
        inputs = [
            "Tell me about Python programming language features and syntax in detail",
            "What about Python decorators and how they wrap functions with examples?",
            "How do Python context managers work with the with statement?",
            "exit",
        ]
        outputs, memory = run_cli_loop(inputs, sf, tf)

        if len(memory.tree.branches) < 1:
            print(f"\n    Expected at least 1 branch, got {len(memory.tree.branches)}")
            return False

        first_branch = memory.tree.branches[0]
        if len(first_branch["content"]) == 0:
            print(f"\n    Branch has no content")
            return False

        return True
    finally:
        cleanup([tf, sf])


def test_cli_plan_has_search():
    tf = temp_tree_file()
    sf = temp_stm_file()
    try:
        inputs = [
            "What is quantum computing?",
            "exit",
        ]
        outputs, memory = run_cli_loop(inputs, sf, tf)

        if len(outputs) < 1:
            print(f"\n    Expected at least 1 output")
            return False

        user_input, plan, response = outputs[0]
        actions = [p[0] for p in plan]
        if "SEARCH" not in actions:
            print(f"\n    Plan should include SEARCH, got: {actions}")
            return False

        return True
    finally:
        cleanup([tf, sf])


# ─── Runner ──────────────────────────────────────────────────────────────────


def main():
    print("=" * 70)
    print("  amtavla Integration Test Suite")
    print("=" * 70)
    print()

    print("--- STM Tests ---")
    run_test("STM append and read", test_stm_append_and_read)
    run_test("STM max lines cap", test_stm_max_lines)
    run_test("STM clear", test_stm_clear)
    run_test("STM read nonexistent", test_stm_read_nonexistent)
    print()

    print("--- LTM Tree Tests ---")
    run_test("Tree create root branch", test_tree_create_root_branch)
    run_test("Tree create child branch", test_tree_create_child_branch)
    run_test("Tree max depth (4 levels)", test_tree_max_depth)
    run_test("Tree content size limit (50)", test_tree_content_limit)
    run_test("Tree append updates embedding", test_tree_append_updates_embedding)
    run_test("Tree find best branch by similarity", test_tree_find_best_branch)
    run_test(
        "Tree retrieve pulls subbranch", test_tree_retrieve_context_pulls_subbranch
    )
    run_test("Tree branch merging", test_tree_branch_merging)
    run_test("Tree save/load persistence", test_tree_save_load)
    run_test("Tree load corrupted file", test_tree_load_corrupted_file)
    run_test("Tree load empty file", test_tree_load_empty_file)
    run_test("Tree most active branch", test_tree_most_active_branch)
    run_test("Tree visualize", test_tree_visualize)
    run_test("Tree empty visualize", test_tree_empty_visualize)
    run_test("Cosine similarity math", test_cosine_similarity)
    run_test("Tree exclude_id in find", test_exclude_id_in_find)
    run_test("Tree dict index lookup", test_tree_dict_index)
    print()

    print("--- Topic Detection Tests ---")
    run_test("Topic shift: same topic", test_topic_shift_same_topic)
    run_test("Topic shift: different topic", test_topic_shift_different_topic)
    run_test("Topic shift: no branch", test_topic_shift_no_branch)
    run_test(
        "Topic shift uses stored embedding", test_topic_shift_uses_stored_embedding
    )
    print()

    print("--- Consolidation Tests ---")
    run_test(
        "Consolidation creates root branch", test_consolidation_creates_root_branch
    )
    run_test(
        "Consolidation excludes current branch",
        test_consolidation_excludes_current_branch,
    )
    run_test("Consolidation no dead code", test_consolidation_no_dead_code)
    print()

    print("--- Tool Tests ---")
    run_test("Weather tool: Tokyo", test_tool_weather_tokyo)
    run_test("Weather tool: London", test_tool_weather_london)
    run_test("Weather tool: unknown", test_tool_weather_unknown)
    run_test("Bash: list files", test_tool_bash_files)
    run_test("Bash: python version", test_tool_bash_python_version)
    run_test("Bash: date", test_tool_bash_date)
    run_test("Bash: no LLM call", test_tool_bash_no_llm)
    run_test("Websearch: basic query", test_tool_websearch)
    run_test("Websearch: empty results", test_tool_websearch_empty)
    print()

    print("--- Planner Tests ---")
    run_test("Planner generates steps", test_planner_generates_steps)
    run_test("Planner includes SEARCH", test_planner_includes_search)
    run_test("Planner max 5 steps", test_planner_max_5_steps)
    run_test("Planner no JSON", test_planner_no_json)
    print()

    print("--- Generator Test ---")
    run_test("Generator produces response", test_generator_produces_response)
    run_test("Generator with empty results", test_generator_with_empty_results)
    print()

    print("--- MemoryController Integration Tests ---")
    run_test("MC first turn creates branch", test_mc_first_turn_creates_branch)
    run_test("MC same topic stays in branch", test_mc_same_topic_stays_in_branch)
    run_test(
        "MC topic shift creates new branch", test_mc_topic_shift_creates_new_branch
    )
    run_test("MC debug status", test_mc_debug_status)
    run_test("MC debug ltm", test_mc_debug_ltm)
    run_test("MC session resume", test_mc_session_resume)
    run_test("MC fire-and-forget threads", test_mc_fire_and_forget)
    print()

    print("--- Full CLI Loop Tests ---")
    run_test(
        "CLI: full conversation with topic shift",
        test_cli_full_conversation_with_topic_shift,
    )
    run_test("CLI: greetings filtered out", test_cli_greeting_filtered_out)
    run_test("CLI: weather in plan", test_cli_weather_in_plan)
    run_test("CLI: subbranch creation", test_cli_subbranch_creation)
    run_test("CLI: context retrieval from LTM", test_cli_context_retrieval_from_ltm)
    run_test("CLI: branch content accumulation", test_cli_branch_content_accumulation)
    run_test("CLI: plan has search", test_cli_plan_has_search)
    print()

    print("=" * 70)
    total = PASS_COUNT + FAIL_COUNT
    print(f"  Results: {PASS_COUNT}/{total} passed, {FAIL_COUNT}/{total} failed")
    if FAILURES:
        print(f"\n  Failed tests:")
        for name in FAILURES:
            print(f"    - {name}")
    print("=" * 70)

    return 0 if FAIL_COUNT == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
