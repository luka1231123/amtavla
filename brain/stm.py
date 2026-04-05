import os
from filelock import FileLock

STM_FILE = "brain/working_memory.txt"
MAX_LINES = 15


def read_stm(stm_file: str | None = None):
    path = stm_file or STM_FILE
    lock = FileLock(f"{path}.lock")
    with lock:
        if not os.path.exists(path):
            return ""
        with open(path, "r") as f:
            return f.read().strip()


def write_stm(content, stm_file: str | None = None):
    path = stm_file or STM_FILE
    lock = FileLock(f"{path}.lock")
    with lock:
        with open(path, "w") as f:
            f.write(content)


def append_stm(snippet, stm_file: str | None = None):
    path = stm_file or STM_FILE
    lock = FileLock(f"{path}.lock")
    with lock:
        lines = []
        if os.path.exists(path):
            with open(path, "r") as f:
                lines = [l for l in f.read().strip().splitlines() if l.strip()]
        lines.append(snippet)
        if len(lines) > MAX_LINES:
            lines = lines[-MAX_LINES:]
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
    print(f"   [DEBUG-STM] -> Appended: {snippet}")


def clear_stm(stm_file: str | None = None):
    path = stm_file or STM_FILE
    lock = FileLock(f"{path}.lock")
    with lock:
        if os.path.exists(path):
            os.remove(path)
            print("   [DEBUG-STM] -> Cleared")
