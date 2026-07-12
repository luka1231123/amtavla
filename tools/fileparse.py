"""FILE_PARSE (T0): read a structured local file into citable text.

Sandboxed to the same read root as NOTE_READ and size-bounded. JSON/CSV/text are
parsed natively; PDF/DOCX are best-effort and degrade to a clear "install X"
error rather than crashing, so the capability is present without pulling heavy
dependencies into the base install.
"""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any

from brain.config import load_brain_config
from brain.memory.catalog import resolve_repo_path
from tools.localfiles import _resolve_within

_MAX_PARSE_BYTES = 2_000_000
_MAX_TEXT_CHARS = 6000
_MAX_CSV_ROWS = 50


class FileParseClient:
    def __init__(self, root: str | Path | None = None):
        if root is None:
            configured = load_brain_config().get("local_files", {}).get("root", "")
            root = resolve_repo_path(configured) if configured else "."
        self.root = Path(root).expanduser().resolve()

    def parse(self, rel_path: str) -> dict[str, Any]:
        path = _resolve_within(self.root, rel_path or "")
        if path is None:
            return {"operation": "parse", "error": f"Path outside allowed root: {rel_path}"}
        if not path.exists() or not path.is_file():
            return {"operation": "parse", "error": f"File not found: {rel_path}"}
        if path.stat().st_size > _MAX_PARSE_BYTES:
            return {"operation": "parse", "error": "File is too large to parse"}

        suffix = path.suffix.lower()
        try:
            if suffix == ".json":
                text = self._parse_json(path)
            elif suffix == ".csv":
                text = self._parse_csv(path)
            elif suffix == ".pdf":
                text = self._parse_pdf(path)
            elif suffix in {".docx"}:
                text = self._parse_docx(path)
            elif suffix in {".txt", ".md", ".log", ".yaml", ".yml", ".ini", ".cfg", ".toml"}:
                text = path.read_text(encoding="utf-8", errors="replace")
            else:
                return {"operation": "parse", "error": f"Unsupported file type: {suffix or 'none'}"}
        except _ParseUnavailable as exc:
            return {"operation": "parse", "path": str(path.relative_to(self.root)), "error": str(exc)}
        except Exception as exc:
            return {"operation": "parse", "path": str(path.relative_to(self.root)), "error": f"Could not parse file: {exc}"}

        truncated = len(text) > _MAX_TEXT_CHARS
        return {
            "operation": "parse",
            "path": str(path.relative_to(self.root)),
            "kind": suffix.lstrip("."),
            "content": text[:_MAX_TEXT_CHARS].strip(),
            "truncated": truncated,
        }

    @staticmethod
    def _parse_json(path: Path) -> str:
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        return json.dumps(data, indent=2, ensure_ascii=False, default=str)

    @staticmethod
    def _parse_csv(path: Path) -> str:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
            reader = csv.reader(handle)
            rows = []
            for index, row in enumerate(reader):
                if index > _MAX_CSV_ROWS:
                    rows.append(f"... ({index}+ rows, truncated)")
                    break
                rows.append(" | ".join(cell.strip() for cell in row))
        return "\n".join(rows)

    @staticmethod
    def _parse_pdf(path: Path) -> str:
        try:
            import pypdf  # type: ignore
        except Exception:
            raise _ParseUnavailable(
                "Reading PDF files needs the 'pypdf' package (pip install pypdf)."
            )
        reader = pypdf.PdfReader(io.BytesIO(path.read_bytes()))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages).strip()

    @staticmethod
    def _parse_docx(path: Path) -> str:
        try:
            import docx  # type: ignore
        except Exception:
            raise _ParseUnavailable(
                "Reading DOCX files needs the 'python-docx' package (pip install python-docx)."
            )
        document = docx.Document(str(path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs).strip()


class _ParseUnavailable(RuntimeError):
    """A parser whose optional dependency is not installed."""
