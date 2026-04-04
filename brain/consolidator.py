import ollama
import json

MODEL = "qwen2.5-coder:1.5b"


def consolidate_stm_to_ltm(working_memory, current_topic):
    print(
        f"   [DEBUG-CONSOLIDATOR] -> Consolidating working memory to LTM (topic: {current_topic})"
    )
    prompt = f"""
You are a memory consolidation agent. Your job is to extract important facts from the current working memory and convert them into structured knowledge for long-term storage.

Working Memory:
{working_memory}

Current Topic: {current_topic}

Extract facts as JSON with this exact format:
{{
  "nodes": [
    {{"id": "unique_id", "label": "Concept Name", "category": "topic/person/preference/fact/project", "strength": 1.0}}
  ],
  "edges": [
    {{"source": "unique_id_1", "target": "unique_id_2", "relation": "describes/depends_on/relates_to/is_a"}}
  ]
}}

Rules:
- Only extract genuinely important facts, not ephemeral details
- Use lowercase snake_case for IDs
- Keep labels concise and human-readable
- Return ONLY valid JSON, no extra text
- Return empty arrays if nothing worth remembering
"""
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    content = response["message"]["content"].strip()
    try:
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        result = json.loads(content)
        print(
            f"   [DEBUG-CONSOLIDATOR] -> Extracted {len(result.get('nodes', []))} nodes, {len(result.get('edges', []))} edges"
        )
        return result
    except json.JSONDecodeError:
        print(f"   [DEBUG-CONSOLIDATOR] -> Failed to parse consolidation JSON")
        return {"nodes": [], "edges": []}


def detect_topic_shift(user_input, current_topic):
    if not current_topic:
        return True, ""
    prompt = f"""
Determine if the user has changed topics.

Current Topic: {current_topic}
New Input: {user_input}

Has the topic changed? Answer with ONLY 'YES' or 'NO' followed by a one-line new topic name if YES.
Example: "YES, user is now asking about weather in Tokyo"
Example: "NO"
"""
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    answer = response["message"]["content"].strip().upper()
    if answer.startswith("YES"):
        new_topic = response["message"]["content"].strip()[3:].strip().rstrip(".")
        print(f"   [DEBUG-CONSOLIDATOR] -> Topic shift detected: '{new_topic}'")
        return True, new_topic
    return False, current_topic


def summarize_for_working_memory(user_input, response):
    prompt = f"""
Create a brief 1-line summary snippet to add to working memory.
Focus on key facts, decisions, or important details.

User: {user_input}
Assistant: {response}

Output ONLY the snippet, nothing else. Keep it under 15 words.
"""
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    snippet = response["message"]["content"].strip()
    print(f"   [DEBUG-CONSOLIDATOR] -> Working memory snippet: {snippet}")
    return snippet
