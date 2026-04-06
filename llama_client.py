import os
import json
import urllib.request
import threading
import time
import hashlib

LLAMA_SERVER_HOST = "127.0.0.1"
LLAMA_SERVER_PORT = 8085
MODEL_PATH = os.path.join(os.path.expanduser("~"), "llama.cpp", "models", "Qwen2.5-Coder-7B-Instruct-F16.gguf")

_server_process = None
_server_lock = threading.Lock()

_response_cache = {}
_cache_lock = threading.Lock()
_CACHE_MAX_SIZE = 100


def _ensure_server_running():
    global _server_process
    with _server_lock:
        if _server_process is not None:
            return
        
        import subprocess
        
        print("Starting llama-server...")
        
        cmd = [
            os.path.expanduser("~/llama.cpp/build/bin/llama-server"),
            "-m", MODEL_PATH,
            "--flash-attn", "on",
            "--host", LLAMA_SERVER_HOST,
            "--port", str(LLAMA_SERVER_PORT),
        ]
        
        _server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        time.sleep(2)
        
        if _server_process.poll() is not None:
            stdout, stderr = _server_process.communicate()
            print(f"llama-server failed to start: {stderr.decode()[:200]}")
            raise RuntimeError(f"llama-server failed to start: {stderr.decode()[:200]}")
        
        for _ in range(30):
            try:
                req = urllib.request.Request(f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}/health")
                urllib.request.urlopen(req, timeout=1)
                print("llama-server ready!")
                return
            except Exception:
                time.sleep(0.2)
        
        raise RuntimeError("llama-server failed to start")


def _stop_server():
    global _server_process
    with _server_lock:
        if _server_process:
            _server_process.terminate()
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
        url,
        data=data,
        headers={"Content-Type": "application/json"}
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
        response = {"message": {"content": content}}
        
        with _cache_lock:
            if len(_response_cache) >= _CACHE_MAX_SIZE:
                _response_cache.clear()
            _response_cache[key] = response
        
        return response
    except Exception as e:
        return {"message": {"content": f"Error: {e}"}}


def embed(prompt: str) -> dict:
    return {"embedding": [0.0] * 768}


def stop():
    _stop_server()