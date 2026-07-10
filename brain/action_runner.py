from __future__ import annotations

import ast
import operator
import re
import time
from typing import Any, Protocol

from brain.contracts import (
    Action,
    ActionResult,
    ActionType,
    ContextPack,
    SearchResult,
    SourceRef,
    stable_source_id,
    utc_now,
)
from tools.websearch import DEFAULT_SEARCH_CLIENT, WebSearchClient


class MemoryActionClient(Protocol):
    def search_memory(self, query: str, top_k: int = 5) -> dict[str, Any]: ...

    def write_memory(self, statement: str) -> dict[str, Any]: ...


_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_MAX_ABS_VALUE = 1_000_000_000_000_000
_EXPLICIT_MEMORY_WRITE = re.compile(
    r"\b(remember|save|store|memorize|note that|don't forget|dont forget|keep (?:this|that|it) in mind)\b",
    re.IGNORECASE,
)


def calculate(expression: str) -> int | float:
    text = (expression or "").strip()
    if not text or len(text) > 200:
        raise ValueError("Calculation must be a non-empty expression under 200 chars")

    tree = ast.parse(text, mode="eval")

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("Only numeric literals are allowed")
            value = node.value
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            left = evaluate(node.left)
            right = evaluate(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 12:
                raise ValueError("Exponent is too large")
            value = _BINARY_OPERATORS[type(node.op)](left, right)
        elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            value = _UNARY_OPERATORS[type(node.op)](evaluate(node.operand))
        else:
            raise ValueError("Only arithmetic expressions are allowed")

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Calculation must produce a real number")
        if abs(value) > _MAX_ABS_VALUE:
            raise ValueError("Calculation result is too large")
        return value

    return evaluate(tree)


class ActionRunner:
    def __init__(
        self,
        *,
        search_client: WebSearchClient | Any | None = None,
        memory_client: MemoryActionClient | None = None,
    ) -> None:
        self.search_client = search_client or DEFAULT_SEARCH_CLIENT
        self.memory_client = memory_client

    def run(
        self,
        action: Action,
        *,
        user_input: str,
        search_cache: dict[str, list[SearchResult]] | None = None,
    ) -> ActionResult:
        started_at = utc_now()
        started = time.perf_counter()
        output: Any = None
        sources: list[SourceRef] = []
        error = ""

        try:
            if action.action_type == ActionType.THINK:
                output = {"instruction": action.detail or "Reason before responding."}
            elif action.action_type == ActionType.SEARCH:
                query = action.detail or user_input
                raw_results = self.search_client.search(query, cache=search_cache)
                results = [
                    item
                    if isinstance(item, SearchResult)
                    else SearchResult.from_row(item, query=query, rank=index)
                    for index, item in enumerate(raw_results, 1)
                ]
                health_check = getattr(self.search_client, "health", None)
                search_health = health_check() if callable(health_check) else {}
                if not results and search_health.get("last_error"):
                    raise RuntimeError(
                        f"Search failed: {search_health['last_error']}"
                    )
                output = list(results)
                sources = [item.to_source() for item in results]
            elif action.action_type == ActionType.CALCULATE:
                expression = action.detail or user_input
                output = {
                    "expression": expression,
                    "value": calculate(expression),
                }
            elif action.action_type == ActionType.MEMORY_SEARCH:
                if self.memory_client is None:
                    raise RuntimeError("Memory actions are unavailable")
                query = action.detail or user_input
                memory_context = self.memory_client.search_memory(query)
                pack = ContextPack.from_memory(memory_context)
                output = pack.to_dict()
                sources = list(pack.sources)
            elif action.action_type == ActionType.MEMORY_WRITE:
                if self.memory_client is None:
                    raise RuntimeError("Memory actions are unavailable")
                if not _EXPLICIT_MEMORY_WRITE.search(user_input or ""):
                    raise PermissionError(
                        "MEMORY_WRITE requires an explicit user request to remember"
                    )
                statement = action.detail or user_input
                item = self.memory_client.write_memory(statement)
                output = item
                memory_item_id = item.get("memory_item_id")
                item_id = memory_item_id or item.get("id")
                if memory_item_id is not None:
                    source_id = f"memory:item:{memory_item_id}"
                elif item_id is not None:
                    source_id = f"memory:semantic:{item_id}"
                else:
                    source_id = stable_source_id("memory:semantic", statement)
                sources = [
                    SourceRef(
                        source_id=source_id,
                        kind="semantic_memory",
                        title=str(item.get("statement") or statement),
                        excerpt=str(item.get("statement") or statement),
                    )
                ]
            else:
                raise ValueError(f"Unsupported action: {action.action_type.value}")
        except Exception as exc:
            error = str(exc)

        return ActionResult(
            action_id=action.action_id,
            action_type=action.action_type,
            detail=action.detail,
            ok=not error,
            output=output,
            sources=sources,
            error=error,
            started_at=started_at,
            completed_at=utc_now(),
            duration_ms=max(0, int((time.perf_counter() - started) * 1000)),
        )
