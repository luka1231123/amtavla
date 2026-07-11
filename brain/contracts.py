from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


def stable_source_id(kind: str, value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{kind}:{digest}"


class ActionType(str, Enum):
    THINK = "THINK"
    SEARCH = "SEARCH"
    CALCULATE = "CALCULATE"
    MEMORY_SEARCH = "MEMORY_SEARCH"
    MEMORY_WRITE = "MEMORY_WRITE"
    SUMMARIZE = "SUMMARIZE"
    REMINDER = "REMINDER"
    NOTE_READ = "NOTE_READ"
    CLARIFY = "CLARIFY"
    RESEARCH = "RESEARCH"

    @classmethod
    def parse(cls, value: str) -> ActionType | None:
        try:
            return cls((value or "").strip().upper())
        except ValueError:
            return None


@dataclass(frozen=True)
class SourceRef:
    source_id: str
    kind: str
    title: str
    excerpt: str = ""
    url: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "kind": self.kind,
            "title": self.title,
            "excerpt": self.excerpt,
            "url": self.url,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SearchResult:
    source_id: str
    title: str
    url: str = ""
    snippet: str = ""
    rank: int = 0
    query: str = ""

    @classmethod
    def from_row(
        cls,
        row: dict[str, Any],
        *,
        query: str,
        rank: int,
    ) -> SearchResult:
        title = str(row.get("title") or "").strip()
        url = str(row.get("url") or row.get("href") or "").strip()
        snippet = str(row.get("snippet") or row.get("body") or "").strip()
        identity = url or f"{query}|{title}|{rank}"
        return cls(
            source_id=stable_source_id("web", identity),
            title=title,
            url=url,
            snippet=snippet,
            rank=rank,
            query=query,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SearchResult:
        query = str(value.get("query") or "")
        rank = int(value.get("rank") or 0)
        fallback = cls.from_row(value, query=query, rank=rank)
        return cls(
            source_id=str(value.get("source_id") or fallback.source_id),
            title=fallback.title,
            url=fallback.url,
            snippet=fallback.snippet,
            rank=rank,
            query=query,
        )

    def to_source(self) -> SourceRef:
        return SourceRef(
            source_id=self.source_id,
            kind="web",
            title=self.title or self.url or "Web result",
            excerpt=self.snippet,
            url=self.url,
            metadata={"rank": self.rank, "query": self.query},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "rank": self.rank,
            "query": self.query,
        }


@dataclass(frozen=True)
class RouteDecision:
    intent: str
    pathway: str
    confidence: float = 0.0
    source: str = "fallback"
    score: int = 0

    @classmethod
    def from_value(cls, value: RouteDecision | dict[str, Any]) -> RouteDecision:
        if isinstance(value, cls):
            return value
        return cls(
            intent=str(value.get("intent") or "default"),
            pathway=str(value.get("pathway") or "planner_full"),
            confidence=float(value.get("confidence") or 0.0),
            source=str(value.get("source") or "fallback"),
            score=int(value.get("score") or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "pathway": self.pathway,
            "confidence": self.confidence,
            "source": self.source,
            "score": self.score,
        }


@dataclass(frozen=True)
class Action:
    action_type: ActionType
    detail: str = ""
    action_id: str = field(default_factory=lambda: new_id("action"))

    @classmethod
    def create(cls, action_type: ActionType | str, detail: str = "") -> Action:
        parsed = (
            action_type
            if isinstance(action_type, ActionType)
            else ActionType.parse(action_type)
        )
        if parsed is None:
            raise ValueError(f"Unsupported action: {action_type}")
        return cls(action_type=parsed, detail=str(detail or "").strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action": self.action_type.value,
            "detail": self.detail,
        }


@dataclass
class Plan:
    actions: list[Action] = field(default_factory=list)
    thinking: str = ""
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_pairs(
        cls,
        steps: list[tuple[str, str]],
        *,
        thinking: str = "",
    ) -> Plan:
        return cls(
            actions=[Action.create(action, detail) for action, detail in steps],
            thinking=thinking,
        )

    def to_pairs(self) -> list[tuple[str, str]]:
        return [(item.action_type.value, item.detail) for item in self.actions]

    def to_dict(self) -> dict[str, Any]:
        return {
            "actions": [item.to_dict() for item in self.actions],
            "thinking": self.thinking,
            "warnings": list(self.warnings),
        }


@dataclass
class ReasoningResult:
    applied: bool = False
    answer_outline: str = ""
    claims: list[dict[str, Any]] = field(default_factory=list)
    uncertainties: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "applied": self.applied,
            "answer_outline": self.answer_outline,
            "claims": [dict(claim) for claim in self.claims],
            "uncertainties": list(self.uncertainties),
            "source_ids": list(self.source_ids),
            "warnings": list(self.warnings),
        }


@dataclass
class ActionResult:
    action_id: str
    action_type: ActionType
    detail: str
    ok: bool
    output: Any = None
    sources: list[SourceRef] = field(default_factory=list)
    error: str = ""
    started_at: str = field(default_factory=utc_now)
    completed_at: str = ""
    duration_ms: int = 0

    @property
    def source_ids(self) -> list[str]:
        return [source.source_id for source in self.sources]

    def to_dict(self) -> dict[str, Any]:
        output = self.output
        if isinstance(output, SearchResult):
            output = output.to_dict()
        elif isinstance(output, list):
            output = [
                item.to_dict() if isinstance(item, SearchResult) else item
                for item in output
            ]
        return {
            "action_id": self.action_id,
            "action": self.action_type.value,
            "detail": self.detail,
            "ok": self.ok,
            "output": output,
            "sources": [source.to_dict() for source in self.sources],
            "source_ids": self.source_ids,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
        }


@dataclass
class ContextPack:
    memory_items: list[dict[str, Any]] = field(default_factory=list)
    semantic_facts: list[dict[str, Any]] = field(default_factory=list)
    episodic_context: list[dict[str, Any]] = field(default_factory=list)
    ltm_context: list[dict[str, Any]] = field(default_factory=list)
    web_results: list[SearchResult] = field(default_factory=list)
    combined_context: str = ""
    style_context: str = ""
    pending_feedback_prompt: str = ""
    pending_feedback_id: int | None = None
    reminder_nudge: str = ""
    sources: list[SourceRef] = field(default_factory=list)

    @classmethod
    def from_memory(cls, value: dict[str, Any]) -> ContextPack:
        memory_items = list(value.get("memory_items") or [])
        semantic = list(value.get("semantic_facts") or value.get("semantic") or [])
        episodic = list(value.get("episodic_context") or value.get("episodic") or [])
        insights = list(value.get("ltm_context") or value.get("insights") or [])
        web_rows = list(value.get("web_results") or [])
        web_results = [
            item if isinstance(item, SearchResult) else SearchResult.from_dict(item)
            for item in web_rows
        ]

        sources: list[SourceRef] = []
        represented_facts = set()
        represented_episodes = set()
        represented_insights = set()
        for item in memory_items:
            item_id = item.get("id")
            content = str(item.get("content") or "")
            metadata = item.get("metadata") or {}
            if metadata.get("legacy_fact_id") is not None:
                represented_facts.add(int(metadata["legacy_fact_id"]))
            if metadata.get("event_id") is not None:
                represented_episodes.add(int(metadata["event_id"]))
            if metadata.get("legacy_insight_id") is not None:
                represented_insights.add(int(metadata["legacy_insight_id"]))
            source_id = (
                f"memory:item:{item_id}"
                if item_id is not None
                else stable_source_id("memory:item", content)
            )
            source_metadata = {
                "item_type": item.get("item_type", "fact"),
                "review_state": item.get("review_state", "candidate"),
                "confidence": item.get("confidence"),
                "updated_at": item.get("updated_at"),
            }
            if item.get("tags"):
                source_metadata["tags"] = [
                    f"{tag['tag_type']}:{tag['name']}" for tag in item["tags"]
                ]
            if metadata.get("stale"):
                source_metadata["stale"] = True
                source_metadata["stale_reason"] = metadata.get("stale_reason", "")
            if metadata.get("contradicts"):
                source_metadata["contradicts"] = list(metadata["contradicts"])
            sources.append(
                SourceRef(
                    source_id=source_id,
                    kind="memory_item",
                    title=f"{item.get('item_type', 'memory')}: {content}",
                    excerpt=content,
                    metadata=source_metadata,
                )
            )
        for item in semantic:
            item_id = item.get("id")
            if item_id is not None and int(item_id) in represented_facts:
                continue
            statement = str(item.get("statement") or "")
            source_id = (
                f"memory:semantic:{item_id}"
                if item_id is not None
                else stable_source_id("memory:semantic", statement)
            )
            sources.append(
                SourceRef(source_id, "semantic_memory", statement, statement)
            )
        for item in episodic:
            item_id = item.get("id")
            if item_id is not None and int(item_id) in represented_episodes:
                continue
            excerpt = (
                f"User: {item.get('user_input', '')} | "
                f"Assistant: {item.get('response', '')}"
            ).strip()
            source_id = (
                f"memory:episode:{item_id}"
                if item_id is not None
                else stable_source_id("memory:episode", excerpt)
            )
            sources.append(
                SourceRef(source_id, "episodic_memory", "Prior turn", excerpt)
            )
        for item in insights:
            item_id = item.get("id")
            if item_id is not None and int(item_id) in represented_insights:
                continue
            thesis = str(item.get("thesis") or "")
            source_id = (
                f"memory:insight:{item_id}"
                if item_id is not None
                else stable_source_id("memory:insight", thesis)
            )
            sources.append(SourceRef(source_id, "insight_memory", thesis, thesis))
        sources.extend(item.to_source() for item in web_results)

        return cls(
            memory_items=memory_items,
            semantic_facts=semantic,
            episodic_context=episodic,
            ltm_context=insights,
            web_results=web_results,
            combined_context=str(value.get("combined_context") or ""),
            style_context=str(value.get("style_context") or ""),
            pending_feedback_prompt=str(value.get("pending_feedback_prompt") or ""),
            pending_feedback_id=value.get("pending_feedback_id"),
            reminder_nudge=str(value.get("reminder_nudge") or ""),
            sources=sources,
        )

    @property
    def source_ids(self) -> list[str]:
        return [source.source_id for source in self.sources]

    def has_context(self) -> bool:
        return bool(
            self.memory_items
            or self.semantic_facts
            or self.episodic_context
            or self.ltm_context
            or self.web_results
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_items": list(self.memory_items),
            "semantic_facts": list(self.semantic_facts),
            "episodic_context": list(self.episodic_context),
            "ltm_context": list(self.ltm_context),
            "web_results": [item.to_dict() for item in self.web_results],
            "combined_context": self.combined_context,
            "style_context": self.style_context,
            "pending_feedback_prompt": self.pending_feedback_prompt,
            "pending_feedback_id": self.pending_feedback_id,
            "reminder_nudge": self.reminder_nudge,
            "sources": [source.to_dict() for source in self.sources],
            "source_ids": self.source_ids,
        }


@dataclass(frozen=True)
class TraceEvent:
    turn_id: str
    event_type: str
    inputs: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, Any] = field(default_factory=dict)
    source_ids: list[str] = field(default_factory=list)
    duration_ms: int = 0
    timestamp: str = field(default_factory=utc_now)
    event_id: str = field(default_factory=lambda: new_id("trace"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "turn_id": self.turn_id,
            "type": self.event_type,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
            "inputs": dict(self.inputs),
            "outputs": dict(self.outputs),
            "source_ids": list(self.source_ids),
        }


@dataclass
class Turn:
    user_input: str
    session_id: str = "default"
    input_source: str = "cli"
    turn_id: str = field(default_factory=lambda: new_id("turn"))
    created_at: str = field(default_factory=utc_now)
    status: str = "received"
    route: RouteDecision | None = None
    context: ContextPack = field(default_factory=ContextPack)
    plan: Plan = field(default_factory=Plan)
    action_results: list[ActionResult] = field(default_factory=list)
    reasoning: ReasoningResult = field(default_factory=ReasoningResult)
    response: str = ""
    response_source_ids: list[str] = field(default_factory=list)
    trace: list[TraceEvent] = field(default_factory=list)
    error: str = ""

    def add_trace(self, event: TraceEvent) -> None:
        self.trace.append(event)

    def to_memory_trace(self) -> dict[str, Any]:
        route = self.route or RouteDecision("default", "planner_full")
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "input_source": self.input_source,
            "intent": route.intent,
            "pathway": route.pathway,
            "todo": [action.to_dict() for action in self.plan.actions],
            "context": self.context.to_dict(),
            "actions": [result.to_dict() for result in self.action_results],
            "reasoning": self.reasoning.to_dict(),
            "source_ids": list(self.response_source_ids),
            "trace": [event.to_dict() for event in self.trace],
            "error": self.error,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "input_source": self.input_source,
            "created_at": self.created_at,
            "status": self.status,
            "user_input": self.user_input,
            "route": self.route.to_dict() if self.route else None,
            "context": self.context.to_dict(),
            "plan": self.plan.to_dict(),
            "action_results": [item.to_dict() for item in self.action_results],
            "reasoning": self.reasoning.to_dict(),
            "response": self.response,
            "response_source_ids": list(self.response_source_ids),
            "trace": [event.to_dict() for event in self.trace],
            "error": self.error,
        }
