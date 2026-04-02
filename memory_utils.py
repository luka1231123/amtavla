import os

MEMORY_FILE = "stm.log"

def read_memory():
    if not os.path.exists(MEMORY_FILE):
        return "Nothing. This is a fresh start."
    with open(MEMORY_FILE, "r") as f:
        return f.read().strip()

def write_memory(content):
    with open(MEMORY_FILE, "w") as f:
        f.write(content)

def delete_memory():
    if os.path.exists(MEMORY_FILE):
        os.remove(MEMORY_FILE)
        print("   [DEBUG-MEM] -> Context switched. Memory wiped.")

def manage_memory(user_input, current_memory):
    """Determines if we are continuing or starting over."""
    if current_memory == "Nothing. This is a fresh start.":
        return # Keep current memory as is
    
    check_prompt = f"""
    Current Task Memory: {current_memory}
    New User Input: {user_input}
    
    Is the user continuing the current task or starting a completely new topic?
    Output ONLY 'CONTINUE' or 'NEW'.
    """
    
    response = ollama.chat(
        model='qwen2.5-coder:1.5b',
        messages=[{'role': 'user', 'content': check_prompt}]
    )
    
    decision = response['message']['content'].strip().upper()
    
    if "NEW" in decision:
        delete_memory()
    else:
        # Update memory with a concise summary of the ongoing thread
        update_prompt = f"Summarize what we are doing in 10 words or less. Old memory: {current_memory}. New input: {user_input}"
        summary = ollama.chat(model='qwen2.5-coder:1.5b', messages=[{'role': 'user', 'content': update_prompt}])
        write_memory(summary['message']['content'].strip())
