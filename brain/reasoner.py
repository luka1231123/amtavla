from __future__ import annotations

import re
from typing import Any

from brain.contracts import (
    ActionResult,
    ActionType,
    ContextPack,
    Plan,
    ReasoningResult,
    RouteDecision,
)
from brain.json_utils import extract_first_json_object
from brain.prompt_builder import PromptBuilder
from generator import build_response_context, collect_response_sources


_QUESTION_RE = re.compile(
    r"(?:\?|^(?:what|which|who|when|where|why|how|should|can|could|would|"
    r"explain|compare|recommend)\b)",
    re.IGNORECASE,
)

_EVIDENCE_ACTIONS = {
    ActionType.SEARCH,
    ActionType.MEMORY_SEARCH,
    ActionType.NOTE_READ,
}


class GroundedReasoner:
    """Build a bounded evidence synthesis after tools and before final writing."""

    def __init__(
        self,
        client: Any | None = None,
        prompt_builder: PromptBuilder | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.prompt_builder = prompt_builder or PromptBuilder()
        settings = (config or {}).get("reasoning", {})
        self.enabled = bool(settings.get("enabled", False))
        self.min_sources = max(1, int(settings.get("min_sources", 2)))
        self.max_context_chars = max(
            1000, min(12000, int(settings.get("max_context_chars", 6500)))
        )

    def _client(self):
        if self.client is not None:
            return self.client
        import llama_client

        return llama_client

    def should_run(
        self,
        user_input: str,
        plan: Plan,
        action_results: list[ActionResult],
        context: ContextPack,
        route: RouteDecision,
    ) -> bool:
        if not self.enabled or not _QUESTION_RE.search(user_input or ""):
            return False
        if route.pathway in {
            "direct_reply",
            "creative_reply",
            "remember_reply",
            "reminder_reply",
            "summarize_reply",
            "research_reply",
        }:
            return False
        successful_evidence_action = any(
            result.ok
            and result.action_type in _EVIDENCE_ACTIONS
            and bool(result.sources)
            for result in action_results
        )
        if successful_evidence_action:
            return True
        source_count = len(collect_response_sources(context, action_results))
        return source_count >= self.min_sources

    def reason(
        self,
        user_input: str,
        plan: Plan,
        action_results: list[ActionResult],
        context: ContextPack,
        route: RouteDecision,
    ) -> ReasoningResult:
        if not self.should_run(user_input, plan, action_results, context, route):
            return ReasoningResult()

        available_sources = collect_response_sources(context, action_results)
        available_ids = {source.source_id for source in available_sources}
        evidence = build_response_context(plan, action_results, context)[
            : self.max_context_chars
        ]
        prompt = self.prompt_builder.build_reasoner_prompt(
            user_input=user_input,
            evidence_context=evidence,
        )
        try:
            response = self._client().chat([{"role": "user", "content": prompt}])
            raw = str((response.get("message") or {}).get("content") or "")
            data = extract_first_json_object(raw)
        except Exception as exc:
            return ReasoningResult(
                applied=True,
                warnings=[f"Reasoning pass failed: {exc}"],
            )
        if not isinstance(data, dict):
            return ReasoningResult(
                applied=True,
                warnings=["Reasoning pass returned invalid JSON"],
            )

        claims = []
        used_ids: list[str] = []
        for raw_claim in data.get("claims") or []:
            if not isinstance(raw_claim, dict):
                continue
            text = " ".join(str(raw_claim.get("text") or "").split()).strip()
            ids = [
                str(source_id)
                for source_id in (raw_claim.get("source_ids") or [])
                if str(source_id) in available_ids
            ]
            if not text:
                continue
            claims.append({"text": text, "source_ids": ids})
            for source_id in ids:
                if source_id not in used_ids:
                    used_ids.append(source_id)

        uncertainties = [
            " ".join(str(item).split())
            for item in (data.get("uncertainties") or [])
            if str(item).strip()
        ][:4]
        outline = " ".join(str(data.get("answer_outline") or "").split()).strip()
        return ReasoningResult(
            applied=True,
            answer_outline=outline,
            claims=claims[:8],
            uncertainties=uncertainties,
            source_ids=used_ids,
        )
