import ollama

def classify_intent(user_prompt, memory):
    routing_instructions = f"""
    Classification Task.
    Context Memory: {memory}
    
    Categories:
    - WEATHER: Weather, temperature, or climate.
    - BASH: Operating system, files, network, or terminal.
    - CHAT: General talk or questions.
    
    Output ONLY the category name.
    """
    
    response = ollama.chat(
        model='qwen2.5-coder:1.5b',
        messages=[
            {'role': 'system', 'content': routing_instructions},
            {'role': 'user', 'content': user_prompt}
        ]
    )
    
    intent = response['message']['content'].strip().upper()
    return intent if intent in ["WEATHER", "BASH", "CHAT"] else "CHAT"
