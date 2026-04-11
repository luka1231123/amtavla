from __future__ import annotations

import re
from pathlib import Path
from typing import Any


class PromptTemplateError(RuntimeError):
    pass


class PromptBuilder:
    _TOKEN_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")

    DEFAULT_PATHWAY_SKILLS = {
        "search_then_reply": ("skills/web_search.md",),
        "remember_reply": ("skills/memory_ops.md",),
        "memory_recall_reply": ("skills/memory_ops.md",),
    }

    def __init__(
        self,
        prompts_root: str | Path | None = None,
        pathway_skill_map: dict[str, tuple[str, ...]] | None = None,
        intent_skill_map: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        repo_root = Path(__file__).resolve().parent.parent
        self.root = (
            Path(prompts_root) if prompts_root else repo_root / "brain" / "prompts"
        ).resolve()
        self.pathway_skill_map = pathway_skill_map or dict(self.DEFAULT_PATHWAY_SKILLS)
        self.intent_skill_map = intent_skill_map or {}

    def build(
        self,
        *,
        task_file: str,
        variables: dict[str, Any] | None = None,
        schema_files: list[str] | None = None,
        intent: str | None = None,
        pathway: str | None = None,
        extra_files: list[str] | None = None,
    ) -> str:
        variables = variables or {}
        schema_files = schema_files or []
        extra_files = extra_files or []

        files: list[str] = ["base/persona.md", task_file]
        files.extend(self._resolve_skill_files(intent=intent, pathway=pathway))
        files.extend(extra_files)
        files.extend(self._normalize_schema_files(schema_files))

        chunks = [self._read_text(rel_path) for rel_path in self._dedupe(files)]
        template = "\n\n".join(chunk.strip() for chunk in chunks if chunk.strip())
        return self._render(template, variables).strip()

    def build_planner_prompt(
        self,
        *,
        user_input: str,
        memory_context: str,
        intent: str | None = None,
        pathway: str | None = None,
    ) -> str:
        context_excerpt = (memory_context or "No prior context.")[:500]
        return self.build(
            task_file="tasks/planner.md",
            variables={
                "user_input": user_input,
                "memory_context": context_excerpt,
            },
            schema_files=["plan_schema.md"],
            intent=intent,
            pathway=pathway,
            extra_files=[
                "policies/internal_usage.md",
                "policies/caveman_internal.md",
            ],
        )

    def build_generator_prompt(
        self,
        *,
        assembled_context: str,
        intent: str | None = None,
        pathway: str | None = None,
    ) -> str:
        return self.build(
            task_file="tasks/generator.md",
            variables={"assembled_context": assembled_context.strip()},
            intent=intent,
            pathway=pathway,
        )

    def build_intent_router_prompt(
        self,
        *,
        user_input: str,
        intents_json: str,
    ) -> str:
        return self.build(
            task_file="tasks/intent_router.md",
            variables={
                "user_input": user_input,
                "available_intents_json": intents_json,
            },
            schema_files=["intent_route_schema.md"],
            extra_files=[
                "policies/internal_usage.md",
                "policies/caveman_internal.md",
            ],
        )

    def render_file(
        self, rel_path: str, variables: dict[str, Any] | None = None
    ) -> str:
        variables = variables or {}
        template = self._read_text(rel_path)
        return self._render(template, variables).strip()

    def _normalize_schema_files(self, schema_files: list[str]) -> list[str]:
        normalized = []
        for name in schema_files:
            if name.startswith("schemas/"):
                normalized.append(name)
            else:
                normalized.append(f"schemas/{name}")
        return normalized

    def _resolve_skill_files(
        self, *, intent: str | None, pathway: str | None
    ) -> list[str]:
        out: list[str] = []
        if pathway and pathway in self.pathway_skill_map:
            out.extend(self.pathway_skill_map[pathway])
        if intent and intent in self.intent_skill_map:
            out.extend(self.intent_skill_map[intent])
        return self._dedupe(out)

    def _read_text(self, rel_path: str) -> str:
        path = (self.root / rel_path).resolve()
        if not self._is_within_root(path):
            raise PromptTemplateError(f"Invalid prompt path outside root: {rel_path}")
        if not path.exists():
            raise PromptTemplateError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8")

    def _is_within_root(self, path: Path) -> bool:
        return path == self.root or self.root in path.parents

    def _render(self, template: str, variables: dict[str, Any]) -> str:
        def repl(match: re.Match[str]) -> str:
            key = match.group(1)
            value = variables.get(key, "")
            return str(value)

        return self._TOKEN_RE.sub(repl, template)

    @staticmethod
    def _dedupe(items: list[str]) -> list[str]:
        seen = set()
        out = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out
