import os
import json
import urllib.request
import threading
import time
import hashlib
import atexit

LLAMA_SERVER_HOST = "127.0.0.1"
LLAMA_SERVER_PORT = 8085
_MODELS_DIR = os.path.join(os.path.expanduser("~"), "llama.cpp", "models")
# Default filename used only when config does not name a model. Kept as the
# current model so behavior is unchanged until the config is pointed at a new
# one; the model swap is a one-line config edit (llm.model_filename).
_DEFAULT_MODEL_FILENAME = "Qwen2.5-Coder-7B-Instruct-Q6_K_L.gguf"
LLAMA_SERVER_BIN = os.path.expanduser("~/llama.cpp/build/bin/llama-server")

_server_process = None
_server_lock = threading.Lock()

_response_cache = {}
_cache_lock = threading.Lock()
_CACHE_MAX_SIZE = 100
_atexit_registered = False


def _llm_config() -> dict:
    """LLM runtime settings from brain_config.json ('llm' block).

    Imported lazily so a missing/partial config never breaks the client, and so
    tests can monkeypatch the config loader.
    """
    try:
        from brain.config import load_brain_config

        cfg = load_brain_config().get("llm", {})
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def resolve_model_path(config: dict | None = None) -> str | None:
    """Resolve the GGUF to load.

    Precedence: an explicit config filename/path, then the historical default
    if present, then the first .gguf in the models dir. The explicit-filename
    path exists so that once a second model is downloaded the process does not
    silently pick an arbitrary file — the swap must be a deliberate config edit.
    """
    cfg = config if config is not None else _llm_config()

    configured_path = (cfg.get("model_path") or "").strip()
    if configured_path:
        # An explicit full path either exists or the model is unavailable —
        # never silently substitute a different file.
        return configured_path if os.path.exists(configured_path) else None

    configured_name = (cfg.get("model_filename") or "").strip()
    if configured_name:
        candidate = os.path.join(_MODELS_DIR, configured_name)
        # Same rule for an explicit filename: missing means degraded, not a
        # guess. Silently picking an arbitrary GGUF once a second model is
        # downloaded is exactly the footgun we are closing.
        return candidate if os.path.exists(candidate) else None

    # No explicit config: convenience fallback for a fresh checkout.
    default = os.path.join(_MODELS_DIR, _DEFAULT_MODEL_FILENAME)
    if os.path.exists(default):
        return default
    if not os.path.isdir(_MODELS_DIR):
        return None
    # Deterministic (sorted) so behavior is stable across runs rather than
    # dependent on directory iteration order.
    for name in sorted(os.listdir(_MODELS_DIR)):
        if name.endswith(".gguf") and not name.startswith("ggml-vocab-"):
            return os.path.join(_MODELS_DIR, name)
    return None


# Back-compat alias for existing callers/tests.
def _resolve_model_path() -> str | None:
    return resolve_model_path()


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
        cmd = _build_server_cmd(model_path)

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


def _build_server_cmd(model_path: str, config: dict | None = None) -> list[str]:
    """Assemble the llama-server launch command from config.

    Pure/deterministic so the exact flags can be asserted in tests without
    spawning a process.
    """
    cfg = config if config is not None else _llm_config()
    cmd = [
        LLAMA_SERVER_BIN,
        "-m",
        model_path,
        "-ngl",
        str(cfg.get("gpu_layers", 99)),
        "-c",
        str(cfg.get("context_size", 4096)),
        "--host",
        LLAMA_SERVER_HOST,
        "--port",
        str(LLAMA_SERVER_PORT),
    ]
    if cfg.get("use_jinja"):
        # Correct model-native chat template; required by Qwen3.x thinking mode.
        cmd.append("--jinja")
    reasoning_mode = str(cfg.get("reasoning_mode", "auto")).lower()
    if reasoning_mode in {"on", "off", "auto"}:
        cmd.extend(["--reasoning", reasoning_mode])
    return cmd


def _sampling_for(profile: str | None, config: dict | None = None) -> dict:
    """Sampling params for a named profile ('default', 'thinking', ...)."""
    cfg = config if config is not None else _llm_config()
    profiles = cfg.get("sampling", {}) if isinstance(cfg.get("sampling"), dict) else {}
    base = {"temperature": 0.7, "max_tokens": cfg.get("max_tokens", 512)}
    base.update(profiles.get("default", {}) if isinstance(profiles.get("default"), dict) else {})
    name = (profile or "default").strip()
    if name != "default" and isinstance(profiles.get(name), dict):
        base.update(profiles[name])
    return base


def _build_payload(
    messages: list[dict],
    *,
    model: str | None = None,
    schema: dict | None = None,
    profile: str | None = None,
    config: dict | None = None,
) -> dict:
    """Build the /v1/chat/completions request body.

    When `schema` is provided it is sent as an OpenAI-style json_schema
    response_format, which llama.cpp converts to a GBNF grammar so malformed
    JSON is impossible at the sampler — not merely discouraged by the prompt.
    """
    sampling = _sampling_for(profile, config=config)
    payload: dict = {
        "model": model or "default",
        "messages": messages,
        "max_tokens": int(sampling.pop("max_tokens", 512)),
    }
    payload.update(sampling)
    if schema is not None:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "structured_output", "schema": schema},
        }
    return payload


def _call_llama(
    messages: list[dict],
    model: str | None = None,
    *,
    schema: dict | None = None,
    profile: str | None = None,
) -> dict:
    _ensure_server_running()

    url = f"http://{LLAMA_SERVER_HOST}:{LLAMA_SERVER_PORT}/v1/chat/completions"
    payload = _build_payload(messages, model=model, schema=schema, profile=profile)

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


def _make_cache_key(
    messages: list[dict],
    model: str | None = None,
    schema: dict | None = None,
    profile: str | None = None,
) -> str:
    cache_str = json.dumps(
        {
            "model": model or "default",
            "messages": messages,
            "schema": schema,
            "profile": profile,
        },
        sort_keys=True,
    )
    return hashlib.sha256(cache_str.encode()).hexdigest()[:32]


def chat(
    messages: list[dict],
    model: str = None,
    *,
    schema: dict | None = None,
    profile: str | None = None,
) -> dict:
    key = _make_cache_key(messages, model=model, schema=schema, profile=profile)

    with _cache_lock:
        if key in _response_cache:
            return _response_cache[key]

    try:
        result = _call_llama(messages, model=model, schema=schema, profile=profile)
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
