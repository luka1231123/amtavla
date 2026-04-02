import ollama

def tool_weather(user_prompt, memory):
    # Memory helps tool if user says "What about Tokyo?" without saying "weather"
    combined_query = (user_prompt + " " + memory).lower()
    if "tokyo" in combined_query:
        return "Tokyo: Sunny, 22C."
    elif "london" in combined_query:
        return "London: Raining, 14C."
    return "Location unknown."

def tool_bash_simulator(user_prompt, memory):
    translation_instructions = f"""
    Context: {memory}
    Task: Translate user request to BASH. 
    Rules: Raw command only. No banter. No markdown. No json.
    """
    
    response = ollama.chat(
        model='qwen2.5-coder:1.5b',
        messages=[
            {'role': 'system', 'content': translation_instructions},
            {'role': 'user', 'content': user_prompt}
        ]
    )
    bash_command = response['message']['content'].strip()
    print(f"   [DEBUG-TOOL] -> Bash Translation: `{bash_command}`")
    
    return f"STDOUT for `{bash_command}`: \nfile1.txt, hidden_folder, system_log.csv"
