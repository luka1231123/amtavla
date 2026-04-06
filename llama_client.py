import os
import json
import urllib.request
import threading
import time
import hashlib
import atexit

LLAMA_SERVER_HOST = "127.0.0.1"
LLAMA_SERVER_PORT = 8085
MODEL_PATH = os.path.join(
    os.path.expanduser("~"),
    "llama.cpp",
    "models",
    "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
)
LLAMA_SERVER_BIN = os.path.expanduser("~/llama.cpp/build/bin/llama-server")

_server_process = None
_server_lock = threading.Lock()

_response_cache = {}
_cache_lock = threading.Lock()
_CACHE_MAX_SIZE = 100
_atexit_registered = False


def _resolve_model_path() -> str | None:
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    model_dir = os.path.join(os.path.expanduser("~"), "llama.cpp", "models")
    if not os.path.isdir(model_dir):
        return None
    for filename in os.listdir(model_dir):
        if filename.endswith(".gguf"):
            return os.path.join(model_dir, filename)
    return None


def _can_use_llama_server() -> tuple[bool, str | None, str | None]:
    if not os.path.exists(LLAMA_SERVER_BIN):
        return False, None, "llama-server binary not found"
    model_path = _resolve_model_path()
    if model_path is None:
        return False, None, "gguf model not found"
    return True, model_path, None


def _ensure_server_running():
    global _server_process, _atexit_registered
    with _server_lock:
        can_use_server, model_path, reason = _can_use_llama_server()
        if not can_use_server:
            raise RuntimeError(f"llama-server unavailable ({reason})")

        if _server_process is not None and _server_process.poll() is None:
            return

        if _server_process is not None and _server_process.poll() is not None:
            _server_process = None

        import subprocess

        print("Starting llama-server...")

        cmd = [
            LLAMA_SERVER_BIN,
            "-m",
            model_path,
            "--host",
            LLAMA_SERVER_HOST,
            "--port",
            str(LLAMA_SERVER_PORT),
        ]

        _server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        if not _atexit_registered:
            atexit.register(_stop_server)
            _atexit_registered = True

        time.sleep(2)

        if _server_process.poll() is not None:
            _server_process = None
            raise RuntimeError("llama-server failed to start")

        for _ in range(30):
            try:
                req = urllib.request.Request(
                    f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}/health"
                )
                urllib.request.urlopen(req, timeout=1)
                print("llama-server ready!")
                return
            except Exception:
                time.sleep(0.2)

        _stop_server()
        raise RuntimeError("llama-server health check failed")


def _stop_server():
    global _server_process
    with _server_lock:
        if _server_process:
            _server_process.terminate()
            try:
                _server_process.wait(timeout=3)
            except Exception:
                _server_process.kill()
            _server_process = None


def _call_llama(messages: list[dict]) -> dict:
    _ensure_server_running()

    url = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}/v1/chat/completions"

    payload = {
        "model": "default",
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.7,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read())
            return result
    except Exception as e:
        raise RuntimeError(f"llama-server call failed: {e}")


def _make_cache_key(messages: list[dict]) -> str:
    cache_str = json.dumps(messages, sort_keys=True)
    return hashlib.sha256(cache_str.encode()).hexdigest()[:32]


def chat(messages: list[dict], model: str = None) -> dict:
    key = _make_cache_key(messages)

    with _cache_lock:
        if key in _response_cache:
            return _response_cache[key]

    try:
        result = _call_llama(messages)
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        if not content:
            content = result.get("message", {}).get("content", "")
        response = {"message": {"content": content}}

        with _cache_lock:
            if len(_response_cache) >= _CACHE_MAX_SIZE:
                _response_cache.clear()
            _response_cache[key] = response

        return response
    except Exception as e:
        return {"message": {"content": f"Error: {e}"}}


def embed(prompt: str) -> dict:
    try:
        import ollama

        response = ollama.embed(model="nomic-embed-text", input=prompt)
        embedding = response.get("embedding")
        if isinstance(embedding, list) and embedding:
            return {"embedding": embedding}
        embeddings = response.get("embeddings", [])
        if embeddings and isinstance(embeddings[0], list):
            return {"embedding": embeddings[0]}
    except Exception:
        pass
    return {"embedding": [0.0] * 768}


def stop():
    _stop_server()
