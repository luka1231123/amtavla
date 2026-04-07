import json
import os
from functools import lru_cache


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "brain_config.json")


@lru_cache(maxsize=1)
def load_brain_config(path: str | None = None) -> dict:
    target = path or DEFAULT_CONFIG_PATH
    if not os.path.exists(target):
        return {}
    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        return {}
    return {}
