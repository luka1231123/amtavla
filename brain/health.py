from __future__ import annotations

from typing import Any

from tools.websearch import DEFAULT_SEARCH_CLIENT


class HealthReporter:
    def __init__(self, memory_client: Any, search_client: Any | None = None) -> None:
        self.memory_client = memory_client
        self.search_client = search_client or DEFAULT_SEARCH_CLIENT

    def _model_health(self) -> dict[str, Any]:
        try:
            import llama_client

            configured, model_path, reason = llama_client._can_use_llama_server()
            process = getattr(llama_client, "_server_process", None)
            running = process is not None and process.poll() is None
            available = configured and running
            return {
                "available": available,
                "configured": configured,
                "running": running,
                "model_path": model_path or "",
                "last_error": reason
                or ("llama-server is not running" if not running else ""),
            }
        except Exception as exc:
            return {
                "available": False,
                "configured": False,
                "running": False,
                "model_path": "",
                "last_error": str(exc),
            }

    def _embedding_health(self) -> dict[str, Any]:
        try:
            service = getattr(self.memory_client, "memory", self.memory_client)
            status = service.get_status()
            available = status.get("embedding_available")
            return {
                "available": available,
                "state": (
                    "healthy"
                    if available is True
                    else "unavailable"
                    if available is False
                    else "unknown"
                ),
                "last_error": status.get("embedding_last_error", ""),
            }
        except Exception as exc:
            return {
                "available": False,
                "state": "unavailable",
                "last_error": str(exc),
            }

    def snapshot(self) -> dict[str, Any]:
        try:
            search = self.search_client.health()
        except Exception as exc:
            search = {"available": False, "last_error": str(exc)}
        return {
            "model": self._model_health(),
            "search": search,
            "embedding": self._embedding_health(),
        }


def render_health(health: dict[str, Any]) -> str:
    lines = ["=== System Health ==="]
    for name in ("model", "search", "embedding"):
        state = health.get(name, {})
        available = state.get("available")
        label = "unknown" if available is None else "ok" if available else "unavailable"
        detail = state.get("last_error") or ""
        line = f"{name.title()}: {label}"
        if detail:
            line += f" ({detail})"
        lines.append(line)
    return "\n".join(lines)
