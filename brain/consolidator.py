import ollama
import json

MODEL = "qwen2.5-coder:1.5b"

MARKERS = ["TASK", "CONTEXT", "IMPORTANT", "FACT", "DECISION", "PREFERENCE"]


def summarize_for_stm(user_input, response):
    prompt = f"""
Summarize this conversation turn into ONE line for short-term memory.
Choose the best marker from: {MARKERS}

Available markers:
[TASK] - what the user wants done
[CONTEXT] - situational background info
[IMPORTANT] - preferences, constraints, critical info
[FACT] - factual claims or concrete details
[DECISION] - agreements or conclusions reached

User: {user_input}
Assistant: {response}

Output format: [MARKER] summary text
Keep it under 20 words. Output ONLY the line.
"""
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    snippet = response["message"]["content"].strip()
    print(f"   [DEBUG-CONSOLIDATOR] -> STM snippet: {snippet}")
    return snippet


def detect_topic_shift(user_input, current_topic):
    if not current_topic:
        return True, ""
    prompt = f"""
Determine if the user has completely changed the subject.

Current Topic: {current_topic}
New Input: {user_input}

Rules:
- If the new input is a follow-up question, agreement, or continuation of the current topic, answer NO.
- ONLY answer YES if the user introduces a drastically different subject.
- if you're unsure, err on the side of NO to avoid false positives.

Has the topic changed? Answer with ONLY 'YES' or 'NO' followed by a one-line new topic name if YES.
"""
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    answer = response["message"]["content"].strip().upper()
    if answer.startswith("YES"):
        new_topic = response["message"]["content"].strip()[3:].strip().rstrip(".")
        print(f"   [DEBUG-CONSOLIDATOR] -> Topic shift: '{new_topic}'")
        return True, new_topic
    return False, current_topic


def promote_stm_to_ltm(stm_lines, current_topic):
    print(f"   [DEBUG-CONSOLIDATOR] -> Promoting STM to LTM (topic: {current_topic})")
    prompt = f"""
Review these short-term memory lines and decide which ones are important enough for long-term storage.

STM Lines:
{stm_lines}

Current Topic: {current_topic}

For each line worth remembering long-term, output it as a new line with:
- An appropriate marker from: {MARKERS}
- The content
- An importance score from 1.0 to 10.0

Format: [MARKER] content | importance: X.X

Rules:
- Only promote genuinely important facts, not ephemeral details
- Skip trivial or already-known information
- Return ONLY the promoted lines, one per line
- Return nothing if nothing is worth keeping
"""
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    content = response["message"]["content"].strip()

    promoted = [l.strip() for l in content.splitlines() if l.strip()]
    print(f"   [DEBUG-CONSOLIDATOR] -> Promoted {len(promoted)} lines to LTM")
    return promoted
