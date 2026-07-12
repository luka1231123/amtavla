"""SHELL_RUN (T2): run one shell command, bounded, and report the result.

This executes a real shell command on the host — which is why it is a T2 action:
it never runs until the user has approved the exact command through the
approvals gate (`brain/approvals.py`). Here we only bound the blast radius that
approval can't: a wall-clock timeout, a captured/­truncated output, and a working
directory. It deliberately does NOT try to sandbox semantics — the human reading
the command before approving is the security boundary.
"""

from __future__ import annotations

import subprocess
from typing import Any

from brain.config import load_brain_config

_DEFAULT_TIMEOUT = 30
_DEFAULT_MAX_OUTPUT = 10_000


class ShellRunner:
    def __init__(
        self,
        *,
        timeout: int | None = None,
        max_output_chars: int | None = None,
        cwd: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        cfg = (config or load_brain_config()).get("tools", {}).get("shell_run", {})
        self.enabled = bool(cfg.get("enabled", False))
        self.timeout = int(timeout if timeout is not None else cfg.get("timeout_seconds", _DEFAULT_TIMEOUT))
        self.max_output_chars = int(
            max_output_chars if max_output_chars is not None
            else cfg.get("max_output_chars", _DEFAULT_MAX_OUTPUT)
        )
        self.cwd = cwd if cwd is not None else (cfg.get("cwd") or None)

    def run(self, command: str) -> dict[str, Any]:
        cmd = (command or "").strip()
        if not cmd:
            return {"operation": "shell_run", "error": "No command given"}
        if not self.enabled:
            return {
                "operation": "shell_run",
                "command": cmd,
                "error": "Shell execution is disabled (set tools.shell_run.enabled).",
            }
        try:
            completed = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.cwd,
            )
        except subprocess.TimeoutExpired:
            return {
                "operation": "shell_run",
                "command": cmd,
                "timed_out": True,
                "error": f"Command timed out after {self.timeout}s",
            }
        except Exception as exc:
            return {"operation": "shell_run", "command": cmd, "error": f"Could not run command: {exc}"}

        stdout, out_trunc = self._cap(completed.stdout)
        stderr, err_trunc = self._cap(completed.stderr)
        return {
            "operation": "shell_run",
            "command": cmd,
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
            "truncated": out_trunc or err_trunc,
            "ok": completed.returncode == 0,
        }

    def _cap(self, text: str) -> tuple[str, bool]:
        text = text or ""
        if len(text) <= self.max_output_chars:
            return text, False
        return text[: self.max_output_chars] + "\n… (output truncated)", True
