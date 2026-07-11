from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable

from brain.action_runner import ActionRunner, reminder_request_supported
from brain.config import load_brain_config
from brain.contracts import (
    Action,
    ActionType,
    ContextPack,
    Plan,
    RouteDecision,
    SearchResult,
    TraceEvent,
    Turn,
)
from brain.health import HealthReporter
from brain.planner import Planner
from brain.reasoner import GroundedReasoner
from generator import (
    ResponseGenerator,
    collect_response_sources,
    render_source_summary,
)

logger = logging.getLogger("brain.orchestrator")

DIRECT_PATHWAYS = {
    "direct_reply",
    "creative_reply",
    "remember_reply",
    "memory_recall_reply",
}

# Pathways that deterministically map to one known action; the planner is not
# consulted, so routing stays debuggable and the small model can't drop the tool.
SINGLE_ACTION_PATHWAYS = {
    "search_then_reply": ActionType.SEARCH,
    "summarize_reply": ActionType.SUMMARIZE,
    "reminder_reply": ActionType.REMINDER,
    "notes_reply": ActionType.NOTE_READ,
    "research_reply": ActionType.RESEARCH,
}


class TurnOrchestrator:
    def __init__(
        self,
        *,
        router: Any,
        memory: Any,
        planner: Any | None = None,
        action_runner: ActionRunner | Any | None = None,
        response_generator: Any | None = None,
        reasoner: Any | None = None,
        health_reporter: Any | None = None,
        debug_hook: Callable[[str, dict[str, Any]], None] | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.router = router
        self.memory = memory
        self.config = config or load_brain_config()
        max_steps = int(self.config.get("routing", {}).get("max_plan_steps", 5))
        self.max_plan_steps = max(1, min(5, max_steps))
        self.planner = planner or Planner(max_steps=self.max_plan_steps)
        self.action_runner = action_runner or ActionRunner(memory_client=memory)
        self.response_generator = response_generator or ResponseGenerator()
        self.reasoner = reasoner or GroundedReasoner(config=self.config)
        self.health_reporter = health_reporter or HealthReporter(
            memory,
            search_client=getattr(self.action_runner, "search_client", None),
        )
        self.debug_hook = debug_hook

    def _record(
        self,
        turn: Turn,
        event_type: str,
        *,
        started: float | None = None,
        inputs: dict[str, Any] | None = None,
        outputs: dict[str, Any] | None = None,
        source_ids: list[str] | None = None,
    ) -> TraceEvent:
        duration_ms = 0
        if started is not None:
            duration_ms = max(0, int((time.perf_counter() - started) * 1000))
        event = TraceEvent(
            turn_id=turn.turn_id,
            event_type=event_type,
            inputs=inputs or {},
            outputs=outputs or {},
            source_ids=source_ids or [],
            duration_ms=duration_ms,
        )
        turn.add_trace(event)
        if self.debug_hook is not None:
            try:
                self.debug_hook(event_type, event.to_dict())
            except Exception:
                logger.debug("Turn debug hook failed", exc_info=True)
        return event

    @staticmethod
    def _brain_dump_mode(user_input: str) -> str:
        text = user_input.lower()
        if "semantic" in text:
            return "semantic"
        if "episodic" in text:
            return "episodic"
        if "insight" in text or "ltm" in text:
            return "insights"
        if "job" in text:
            return "jobs"
        return "full"

    def _bounded_plan(self, plan: Plan) -> Plan:
        if len(plan.actions) <= self.max_plan_steps:
            return plan
        return Plan(
            actions=plan.actions[: self.max_plan_steps],
            thinking=plan.thinking,
            warnings=plan.warnings
            + [f"Plan truncated to {self.max_plan_steps} actions"],
        )

    async def _create_plan(self, turn: Turn) -> Plan:
        route = turn.route or RouteDecision("default", "planner_full")
        if route.pathway in DIRECT_PATHWAYS:
            return Plan()
        if route.pathway in SINGLE_ACTION_PATHWAYS:
            action_type = SINGLE_ACTION_PATHWAYS[route.pathway]
            if action_type == ActionType.REMINDER and not reminder_request_supported(
                turn.user_input
            ):
                # A fuzzy route picked reminder_reply but the REMINDER gate
                # would refuse it; let the planner choose instead of running
                # an action that is guaranteed to fail.
                return await asyncio.to_thread(
                    self.planner.create_plan,
                    turn.user_input,
                    turn.context.combined_context,
                    route.intent,
                    route.pathway,
                )
            return Plan(
                actions=[Action(action_type=action_type, detail=turn.user_input)]
            )
        return await asyncio.to_thread(
            self.planner.create_plan,
            turn.user_input,
            turn.context.combined_context,
            route.intent,
            route.pathway,
        )

    async def process(
        self,
        user_input: str,
        *,
        session_id: str = "default",
        input_source: str = "cli",
    ) -> Turn:
        turn = Turn(
            user_input=(user_input or "").strip(),
            session_id=session_id,
            input_source=input_source,
        )
        commit_queued = False
        foreground_started = False
        memory_response = ""

        try:
            await asyncio.to_thread(self.memory.begin_foreground_turn)
            foreground_started = True
            self._record(
                turn,
                "user_prompt",
                inputs={"text": turn.user_input, "input_source": input_source},
                outputs={"accepted": bool(turn.user_input)},
            )
            if not turn.user_input:
                raise ValueError("User input is empty")

            started = time.perf_counter()
            try:
                health = await asyncio.to_thread(self.health_reporter.snapshot)
            except Exception as exc:
                health = {"available": False, "last_error": str(exc)}
            self._record(turn, "health", started=started, outputs=health)

            started = time.perf_counter()
            raw_route = await asyncio.to_thread(self.router.route, turn.user_input)
            turn.route = RouteDecision.from_value(raw_route)
            self._record(
                turn,
                "intent_decision",
                started=started,
                inputs={"text": turn.user_input},
                outputs=turn.route.to_dict(),
            )
            self._record(
                turn,
                "route_pathway",
                outputs={
                    "intent": turn.route.intent,
                    "pathway": turn.route.pathway,
                },
            )

            if turn.route.pathway == "brain_dump_reply":
                mode = self._brain_dump_mode(turn.user_input)
                started = time.perf_counter()
                turn.response = await asyncio.to_thread(
                    self.memory.get_brain_dump,
                    mode,
                )
                memory_response = turn.response
                self._record(
                    turn,
                    "assistant_response",
                    started=started,
                    inputs={"brain_dump_mode": mode},
                    outputs={"response": turn.response},
                )
            else:
                started = time.perf_counter()
                raw_context = await asyncio.to_thread(
                    self.memory.get_context_for_prompt,
                    turn.user_input,
                    False,
                    turn.route.intent,
                    turn.route.pathway,
                    None,
                )
                turn.context = ContextPack.from_memory(raw_context)
                self._record(
                    turn,
                    "context",
                    started=started,
                    inputs={"query": turn.user_input, "include_web": False},
                    outputs={
                        "semantic_count": len(turn.context.semantic_facts),
                        "episodic_count": len(turn.context.episodic_context),
                        "insight_count": len(turn.context.ltm_context),
                        "web_count": len(turn.context.web_results),
                    },
                    source_ids=turn.context.source_ids,
                )

                started = time.perf_counter()
                turn.plan = self._bounded_plan(await self._create_plan(turn))
                self._record(
                    turn,
                    "plan",
                    started=started,
                    inputs={
                        "intent": turn.route.intent,
                        "pathway": turn.route.pathway,
                    },
                    outputs=turn.plan.to_dict(),
                    source_ids=turn.context.source_ids,
                )

                search_cache: dict[str, list[SearchResult]] = {}
                for action in turn.plan.actions:
                    action_started = time.perf_counter()
                    result = await asyncio.to_thread(
                        self.action_runner.run,
                        action,
                        user_input=turn.user_input,
                        search_cache=search_cache,
                    )
                    turn.action_results.append(result)
                    self._record(
                        turn,
                        "action_result",
                        started=action_started,
                        inputs=action.to_dict(),
                        outputs=result.to_dict(),
                        source_ids=result.source_ids,
                    )

                response_sources = collect_response_sources(
                    turn.context,
                    turn.action_results,
                )

                reasoning_started = time.perf_counter()
                turn.reasoning = await asyncio.to_thread(
                    self.reasoner.reason,
                    turn.user_input,
                    turn.plan,
                    turn.action_results,
                    turn.context,
                    turn.route,
                )
                if turn.reasoning.applied:
                    self._record(
                        turn,
                        "reasoning",
                        started=reasoning_started,
                        inputs={
                            "available_source_ids": [
                                source.source_id for source in response_sources
                            ]
                        },
                        outputs=turn.reasoning.to_dict(),
                        source_ids=turn.reasoning.source_ids,
                    )

                started = time.perf_counter()
                clarify_question = next(
                    (
                        str((result.output or {}).get("question", "")).strip()
                        for result in turn.action_results
                        if result.ok
                        and result.action_type == ActionType.CLARIFY
                        and isinstance(result.output, dict)
                    ),
                    "",
                )
                if clarify_question:
                    # The application, not the model, owns clarification: the
                    # planned question IS the reply, verbatim.
                    base_response = clarify_question
                elif turn.route.intent == "memory_recall" and not turn.context.has_context():
                    # memory_recall_reply is a direct pathway (no planned
                    # actions), so with no recalled context there is nothing
                    # to answer from.
                    base_response = "IDK"
                else:
                    if turn.reasoning.applied:
                        base_response = await asyncio.to_thread(
                            self.response_generator.generate,
                            turn.user_input,
                            turn.plan,
                            turn.action_results,
                            turn.context,
                            turn.route,
                            turn.reasoning,
                        )
                    else:
                        base_response = await asyncio.to_thread(
                            self.response_generator.generate,
                            turn.user_input,
                            turn.plan,
                            turn.action_results,
                            turn.context,
                            turn.route,
                        )
                turn.response = base_response
                memory_response = base_response
                cited_sources = [
                    source
                    for source in response_sources
                    if f"[{source.source_id}]" in base_response
                ]
                turn.response_source_ids = [
                    source.source_id for source in cited_sources
                ]
                show_sources = bool(
                    self.config.get("routing", {}).get("show_sources", True)
                )
                if (
                    show_sources
                    and base_response != "IDK"
                    and not clarify_question
                    and turn.route.pathway not in {"direct_reply", "creative_reply"}
                ):
                    summary = render_source_summary(base_response, cited_sources)
                    if summary:
                        turn.response += "\n\n" + summary
                if turn.context.reminder_nudge:
                    turn.response += "\n\n" + turn.context.reminder_nudge
                # pending_feedback_prompt is NOT appended: the memory-check
                # question is delivered as its own message (with yes/no
                # buttons in the UI) by the transport layer.
                self._record(
                    turn,
                    "assistant_response",
                    started=started,
                    inputs={
                        "action_count": len(turn.action_results),
                        "source_ids": turn.response_source_ids,
                    },
                    outputs={"response": turn.response},
                    source_ids=turn.response_source_ids,
                )

            turn.status = "completed"
            self._record(
                turn,
                "memory_commit",
                inputs={"turn_id": turn.turn_id},
                outputs={"state": "queued"},
                source_ids=turn.response_source_ids,
            )
            await asyncio.to_thread(
                self.memory.process_turn_async,
                turn.user_input,
                memory_response,
                turn.to_memory_trace(),
            )
            commit_queued = True
        except Exception as exc:
            turn.status = "failed"
            turn.error = str(exc)
            if not turn.response:
                turn.response = "I'm having trouble right now. Please try again."
            self._record(
                turn,
                "turn_error",
                outputs={"error": turn.error, "response": turn.response},
            )
            if foreground_started and not commit_queued and turn.user_input:
                try:
                    await asyncio.to_thread(
                        self.memory.process_turn_async,
                        turn.user_input,
                        turn.response,
                        turn.to_memory_trace(),
                    )
                    commit_queued = True
                except Exception:
                    logger.exception("Failed to queue failed turn for memory commit")
        finally:
            if foreground_started:
                await asyncio.to_thread(self.memory.end_foreground_turn)

        return turn
