import ollama

def generate_final_response(user_prompt, context_data):
    generator_instructions = f"Answer naturally and concisely using this data: {context_data}. No yapping."
    response = ollama.chat(
        model='qwen2.5-coder:1.5b',
        messages=[{'role': 'system', 'content': generator_instructions}, {'role': 'user', 'content': user_prompt}]
    )
    return response['message']['content'].strip()
