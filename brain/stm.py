import os

WORKING_MEMORY_FILE = "brain/working_memory.txt"
MAX_LINES = 15


def read_working_memory():
    if not os.path.exists(WORKING_MEMORY_FILE):
        return ""
    with open(WORKING_MEMORY_FILE, "r") as f:
        return f.read().strip()


def write_working_memory(content):
    with open(WORKING_MEMORY_FILE, "w") as f:
        f.write(content)


def append_to_working_memory(snippet):
    lines = []
    if os.path.exists(WORKING_MEMORY_FILE):
        with open(WORKING_MEMORY_FILE, "r") as f:
            lines = [l for l in f.read().strip().splitlines() if l.strip()]
    lines.append(snippet)
    if len(lines) > MAX_LINES:
        lines = lines[-MAX_LINES:]
    with open(WORKING_MEMORY_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"   [DEBUG-STM] -> Appended to working memory: {snippet}")


def clear_working_memory():
    if os.path.exists(WORKING_MEMORY_FILE):
        os.remove(WORKING_MEMORY_FILE)
        print("   [DEBUG-STM] -> Working memory cleared")
