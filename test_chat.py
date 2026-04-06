#!/usr/bin/env python3
"""Quick test for amtavla chat pipeline."""

import sys
import time
import os

sys.path.insert(0, ".")

import llama_client
from brain.planner import generate_plan
from tools.websearch import tool_websearch
from generator import generate_response


def test_llama_chat():
    print("=== Test 1: LLM Chat (warm up + cache) ===")
    
    start = time.time()
    resp1 = llama_client.chat([{"role": "user", "content": "Hello"}])
    t1 = time.time() - start
    print(f"First call (cold): {t1:.2f}s")
    print(f"  Response: {resp1['message']['content'][:50]}...")
    
    start = time.time()
    resp2 = llama_client.chat([{"role": "user", "content": "Hello"}])
    t2 = time.time() - start
    print(f"Second call (cached): {t2:.3f}s")
    print(f"  Cache hit: {t2 < 0.1}")


def test_full_pipeline():
    print("\n=== Test 2: Full Pipeline ===")
    
    user_input = "What is Python?"
    
    # Plan
    start = time.time()
    plan, thinking = generate_plan(user_input, "")
    t_plan = time.time() - start
    print(f"generate_plan: {t_plan:.2f}s -> {plan}")
    
    # Execute plan steps
    start = time.time()
    plan_results = []
    for action, detail in plan:
        if action == "SEARCH":
            result = tool_websearch(detail or user_input)
            plan_results.append((action, detail, result))
        elif action == "THINK":
            plan_results.append((action, detail, thinking))
    t_exec = time.time() - start
    print(f"execute_plan: {t_exec:.2f}s")
    
    # Generate response
    start = time.time()
    response = generate_response(user_input, plan, plan_results, {})
    t_resp = time.time() - start
    print(f"generate_response: {t_resp:.2f}s")
    print(f"  Response: {response[:100]}...")
    
    total = t_plan + t_exec + t_resp
    print(f"\nTotal: {total:.2f}s")


def test_cache():
    print("\n=== Test 3: Cache Behavior ===")
    
    queries = [
        "What is 2+2?",
        "What is 2+2?",
        "What is Python?",
        "What is Python?",
    ]
    
    for i, q in enumerate(queries):
        start = time.time()
        resp = llama_client.chat([{"role": "user", "content": q}])
        t = time.time() - start
        cached = t < 0.1
        print(f"  Query {i+1}: {t:.3f}s (cached: {cached})")


def test_web_search():
    print("\n=== Test 4: Web Search Integration ===")
    
    start = time.time()
    result = tool_websearch("Python programming")
    t = time.time() - start
    print(f"Web search: {t:.2f}s")
    print(f"  Result: {result[:150]}...")


if __name__ == "__main__":
    print("amtavla Quick Test\n")
    print("=" * 40)
    
    test_llama_chat()
    test_full_pipeline()
    test_cache()
    test_web_search()
    
    print("\n" + "=" * 40)
    print("Done!")