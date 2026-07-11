from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from brain.config import load_brain_config
from brain.memory.catalog import resolve_repo_path

_MAX_LIST_ENTRIES = 40
_MAX_READ_BYTES = 8192
_MAX_MATCHES = 10
_SKIP_DIRS = {
    ".git", "__pycache__", "venv", "node_modules", ".idea", ".vscode",
    "logs", "exports",
}
_TEXT_SUFFIXES = {
    ".txt", ".md", ".py", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".csv", ".html", ".css", ".js", ".sh", ".log",
}

_READ_RE = re.compile(r"\b(?:read|open|show|cat)\s+(?:file\s+)?([\w./ -]+\.\w{1,8})", re.IGNORECASE)


class LocalFilesClient:
    """Sandboxed, read-only access to files under one allowlisted root.

    Every operation resolves paths against the root and refuses anything that
    escapes it. Reads are size-bounded and text-only, listings are entry-bounded,
    so a single action can never dump the disk into the context window.
    """

    def __init__(self, root: str | Path | None = None):
        if root is None:
            config = load_brain_config()
            configured = config.get("local_files", {}).get("root", "")
            root = resolve_repo_path(configured) if configured else os.getcwd()
        self.root = Path(root).expanduser().resolve()

    def _safe_path(self, candidate: str) -> Path | None:
        path = (self.root / candidate.strip().lstrip("/")).resolve()
        if path != self.root and self.root not in path.parents:
            return None
        return path

    def run(self, detail: str) -> dict[str, Any]:
        """Interpret a natural or structured detail: read <path>, list [subdir], find <term>."""
        detail = (detail or "").strip()
        read_match = _READ_RE.search(detail)
        if read_match:
            return self.read_file(read_match.group(1).strip())
        lowered = detail.lower()
        if not detail or re.search(r"\b(list|ls|files|directory|folder)\b", lowered):
            subdir_match = re.search(r"\b(?:in|under|inside)\s+([\w./ -]+)", detail)
            return self.list_files(subdir_match.group(1).strip() if subdir_match else "")
        return self.find_files(detail)

    def list_files(self, subdir: str = "") -> dict[str, Any]:
        target = self._safe_path(subdir or ".")
        if target is None or not target.exists():
            return {"operation": "list", "root": str(self.root), "error": f"Directory not found or outside allowed root: {subdir}"}
        if not target.is_dir():
            return self.read_file(str(target.relative_to(self.root)))
        entries = []
        for entry in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            if entry.name.startswith(".") or entry.name in _SKIP_DIRS:
                continue
            entries.append(
                {
                    "name": entry.name + ("/" if entry.is_dir() else ""),
                    "size": entry.stat().st_size if entry.is_file() else None,
                }
            )
            if len(entries) >= _MAX_LIST_ENTRIES:
                break
        return {
            "operation": "list",
            "root": str(self.root),
            "directory": str(target.relative_to(self.root)) or ".",
            "entries": entries,
            "truncated": len(entries) >= _MAX_LIST_ENTRIES,
        }

    def read_file(self, rel_path: str) -> dict[str, Any]:
        path = self._safe_path(rel_path)
        if path is None:
            return {"operation": "read", "error": f"Path outside allowed root: {rel_path}"}
        if not path.exists() or not path.is_file():
            return {"operation": "read", "error": f"File not found: {rel_path}"}
        if path.suffix.lower() not in _TEXT_SUFFIXES:
            return {"operation": "read", "error": f"Only text files can be read: {rel_path}"}
        raw = path.read_bytes()[: _MAX_READ_BYTES + 1]
        truncated = len(raw) > _MAX_READ_BYTES
        text = raw[:_MAX_READ_BYTES].decode("utf-8", errors="replace")
        return {
            "operation": "read",
            "path": str(path.relative_to(self.root)),
            "content": text,
            "truncated": truncated,
        }

    def find_files(self, term: str) -> dict[str, Any]:
        tokens = [t for t in re.findall(r"[a-z0-9]+", term.lower()) if len(t) > 2]
        matches = []
        if tokens:
            for dirpath, dirnames, filenames in os.walk(self.root):
                dirnames[:] = [
                    d for d in dirnames if not d.startswith(".") and d not in _SKIP_DIRS
                ]
                for name in filenames:
                    if name.startswith("."):
                        continue
                    if any(token in name.lower() for token in tokens):
                        rel = str(Path(dirpath, name).relative_to(self.root))
                        matches.append(rel)
                        if len(matches) >= _MAX_MATCHES:
                            break
                if len(matches) >= _MAX_MATCHES:
                    break
        return {
            "operation": "find",
            "root": str(self.root),
            "term": term,
            "matches": matches,
        }
