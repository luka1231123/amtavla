import os

STM_FILE = "brain/working_memory.txt"
MAX_LINES = 15


def read_stm():
    if not os.path.exists(STM_FILE):
        return ""
    with open(STM_FILE, "r") as f:
        return f.read().strip()


def write_stm(content):
    with open(STM_FILE, "w") as f:
        f.write(content)


def append_stm(snippet):
    lines = []
    if os.path.exists(STM_FILE):
        with open(STM_FILE, "r") as f:
            lines = [l for l in f.read().strip().splitlines() if l.strip()]
    lines.append(snippet)
    if len(lines) > MAX_LINES:
        lines = lines[-MAX_LINES:]
    with open(STM_FILE, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"   [DEBUG-STM] -> Appended: {snippet}")


def clear_stm():
    if os.path.exists(STM_FILE):
        os.remove(STM_FILE)
        print("   [DEBUG-STM] -> Cleared")
