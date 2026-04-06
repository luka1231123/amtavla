import json
import os
from datetime import datetime
from filelock import FileLock

STM_FILE = "brain/working_memory.json"
MAX_ENTRIES = 15


def read_stm(stm_file: str | None = None) -> str:
    path = stm_file or STM_FILE
    lock = FileLock(f"{path}.lock")
    with lock:
        if not os.path.exists(path):
            return ""
        try:
            with open(path, "r") as f:
                data = json.load(f)
            entries = data.get("entries", [])
            lines = []
            for e in entries:
                lines.append(f"User: {e.get('user', '')}")
                lines.append(f"Bot: {e.get('response', '')}")
            return "\n".join(lines)
        except (json.JSONDecodeError, IOError):
            return ""


def write_stm(data: dict, stm_file: str | None = None):
    path = stm_file or STM_FILE
    lock = FileLock(f"{path}.lock")
    with lock:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def append_stm(user: str, response: str, stm_file: str | None = None):
    path = stm_file or STM_FILE
    lock = FileLock(f"{path}.lock")
    with lock:
        data = {"entries": []}
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                data = {"entries": []}

        entries = data.get("entries", [])
        entries.append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "user": user,
            "response": response,
        })

        if len(entries) > MAX_ENTRIES:
            entries = entries[-MAX_ENTRIES:]

        data["entries"] = entries

        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def clear_stm(stm_file: str | None = None):
    path = stm_file or STM_FILE
    lock = FileLock(f"{path}.lock")
    with lock:
        if os.path.exists(path):
            os.remove(path)
