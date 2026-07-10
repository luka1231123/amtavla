from __future__ import annotations

import json
from typing import Any

from brain.contracts import (
    ActionResult,
    ActionType,
    ContextPack,
    Plan,
    RouteDecision,
    SearchResult,
    SourceRef,
    utc_now,
)
from brain.prompt_builder import PromptBuilder
from tools.websearch import render_search_results

PROMPT_BUILDER = PromptBuilder()


def _source_for_kind(sources: list[SourceRef], kind: str, index: int) -> str:
    matching = [item.source_id for item in sources if item.kind == kind]
    return matching[index] if index < len(matching) else f"{kind}:{index + 1}"


def _render_action_output(result: ActionResult) -> str:
    if not result.ok:
        return f"Action failed: {result.error}"
    if isinstance(result.output, str):
        return result.output
    if result.action_type == ActionType.SEARCH:
        search_results = []
        for item in result.output or []:
            if isinstance(item, SearchResult):
                search_results.append(item)
            elif isinstance(item, dict):
                search_results.append(SearchResult.from_dict(item))
        return render_search_results(search_results)
    return json.dumps(result.output, indent=2, ensure_ascii=True, default=str)


def collect_response_sources(
    context: ContextPack,
    action_results: list[ActionResult],
) -> list[SourceRef]:
    sources = list(context.sources)
    for result in action_results:
        sources.extend(result.sources)
    deduped = []
    seen = set()
    for source in sources:
        if source.source_id in seen:
            continue
        seen.add(source.source_id)
        deduped.append(source)
    return deduped


def build_response_context(
    plan: Plan,
    action_results: list[ActionResult],
    context: ContextPack,
) -> str:
    parts = []

    if plan.actions:
        lines = [
            f"- {action.action_type.value}: {action.detail}"
            for action in plan.actions
        ]
        parts.append("--- Plan ---\n" + "\n".join(lines))
    if plan.warnings:
        parts.append(
            "--- Plan Validation Warnings ---\n"
            + "\n".join(f"- {warning}" for warning in plan.warnings)
        )

    for result in action_results:
        label = result.action_type.value
        if result.detail:
            label += f" ({result.detail})"
        rendered = _render_action_output(result).strip()
        if rendered:
            parts.append(f"--- {label} ---\n{rendered}")

    if context.memory_items:
        lines = []
        conflict_pairs = []
        item_ids = {
            item.get("id"): index for index, item in enumerate(context.memory_items)
        }
        for index, item in enumerate(context.memory_items):
            source_id = _source_for_kind(context.sources, "memory_item", index)
            item_type = item.get("item_type", "memory")
            state = item.get("review_state", "candidate")
            metadata = item.get("metadata") or {}
            notes = []
            if item.get("tags"):
                notes.append(
                    "tags: "
                    + ", ".join(f"{t['tag_type']}:{t['name']}" for t in item["tags"])
                )
            if metadata.get("stale"):
                notes.append(f"STALE: {metadata.get('stale_reason', 'outdated')}")
            for other_id in metadata.get("contradicts") or []:
                notes.append(f"CONFLICTS with item #{other_id}")
                if other_id in item_ids:
                    conflict_pairs.append((index, item_ids[other_id]))
            note_str = f" [{'; '.join(notes)}]" if notes else ""
            lines.append(
                f"[{source_id}] ({item_type}, {state}){note_str} {item.get('content', '')}"
            )
        parts.append("--- Unified Memory ---\n" + "\n".join(lines))
        if conflict_pairs:
            parts.append(
                "--- Conflicting Memory ---\n"
                "Some supplied memories contradict each other. Do NOT silently pick "
                "one. Tell the user both versions exist, cite both source IDs, and "
                "say which is more recent."
            )

    if context.semantic_facts and not context.memory_items:
        lines = []
        for index, item in enumerate(context.semantic_facts):
            source_id = _source_for_kind(context.sources, "semantic_memory", index)
            lines.append(f"[{source_id}] {item.get('statement', '')}")
        parts.append("--- Semantic Memory ---\n" + "\n".join(lines))

    if context.episodic_context and not context.memory_items:
        lines = []
        for index, item in enumerate(context.episodic_context):
            source_id = _source_for_kind(context.sources, "episodic_memory", index)
            lines.append(
                f"[{source_id}] User: {item.get('user_input', '')} | "
                f"Assistant: {item.get('response', '')}"
            )
        parts.append("--- Episodic Recall ---\n" + "\n".join(lines))

    if context.ltm_context and not context.memory_items:
        lines = []
        for index, item in enumerate(context.ltm_context):
            source_id = _source_for_kind(context.sources, "insight_memory", index)
            lines.append(f"[{source_id}] {item.get('thesis', '')}")
        parts.append("--- Long-Term Insights ---\n" + "\n".join(lines))

    if context.web_results:
        parts.append(
            "--- Web Grounding ---\n" + render_search_results(context.web_results)
        )

    if not (
        context.memory_items
        or context.semantic_facts
        or context.episodic_context
        or context.ltm_context
        or context.web_results
    ) and context.combined_context:
        parts.append("--- Recall Context ---\n" + context.combined_context)

    if context.style_context:
        parts.append(
            "--- User Style Profile ---\n"
            "Apply these standing style rules to your reply:\n"
            + context.style_context
        )

    sources = collect_response_sources(context, action_results)
    if sources:
        lines = []
        for source in sources:
            label = source.title or source.url or source.kind
            lines.append(f"[{source.source_id}] {source.kind}: {label}")
        parts.append("--- Source Catalog ---\n" + "\n".join(lines))

    return "\n\n".join(parts)


def render_source_summary(response: str, sources: list[SourceRef]) -> str:
    """Readable user-facing footer for the sources the answer actually cited.

    Falls back to the top available sources when the model cited none.
    """
    if not sources:
        return ""
    cited = [s for s in sources if s.source_id in (response or "")]
    shown = cited or sources[:3]
    lines = []
    for source in shown:
        if source.kind == "web":
            label = source.url or source.title
        else:
            label = source.excerpt or source.title
        label = " ".join(str(label).split())
        if len(label) > 90:
            label = label[:90].rstrip() + "..."
        kind = source.kind.replace("_", " ")
        flags = []
        if source.metadata.get("stale"):
            flags.append("stale")
        if source.metadata.get("contradicts"):
            flags.append("conflicting")
        flag_str = f" ({', '.join(flags)})" if flags else ""
        lines.append(f"  - {kind}{flag_str}: {label}")
    return "Sources:\n" + "\n".join(lines)


class ResponseGenerator:
    def __init__(
        self,
        client: Any | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self.client = client
        self.prompt_builder = prompt_builder or PROMPT_BUILDER

    def _client(self):
        if self.client is not None:
            return self.client
        import llama_client

        return llama_client

    def generate(
        self,
        user_prompt: str,
        plan: Plan,
        action_results: list[ActionResult],
        context: ContextPack,
        route: RouteDecision,
    ) -> str:
        context_str = build_response_context(plan, action_results, context)
        system_message = self.prompt_builder.build_generator_prompt(
            assembled_context=context_str,
            intent=route.intent,
            pathway=route.pathway,
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self._client().chat(messages)
            text = response["message"]["content"].strip()
            return text or "IDK"
        except Exception as exc:
            return f"I apologize, but I encountered an error generating a response: {exc}"


def _legacy_results(plan_results: list[tuple[str, str, str]]) -> list[ActionResult]:
    results = []
    for action, detail, output in plan_results:
        action_type = ActionType.parse(action) or ActionType.THINK
        results.append(
            ActionResult(
                action_id=f"legacy-{len(results) + 1}",
                action_type=action_type,
                detail=detail,
                ok=not str(output).startswith("Error:"),
                output=output,
                error=str(output)[6:].strip() if str(output).startswith("Error:") else "",
                completed_at=utc_now(),
            )
        )
    return results


def generate_response(
    user_prompt,
    plan,
    plan_results,
    context_blocks,
    intent: str | None = None,
    pathway: str | None = None,
):
    structured_plan = plan if isinstance(plan, Plan) else Plan.from_pairs(plan or [])
    structured_results = (
        plan_results
        if all(isinstance(item, ActionResult) for item in (plan_results or []))
        else _legacy_results(plan_results or [])
    )
    context = (
        context_blocks
        if isinstance(context_blocks, ContextPack)
        else ContextPack.from_memory(context_blocks or {})
    )
    route = RouteDecision(intent or "default", pathway or "planner_full")
    return ResponseGenerator().generate(
        user_prompt,
        structured_plan,
        structured_results,
        context,
        route,
    )
