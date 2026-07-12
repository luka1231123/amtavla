from __future__ import annotations

import ast
import json
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
from brain.memory.commitments import extract_commitments, parse_reminder
from brain.trust import action_tier
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
_EXPLICIT_REMINDER = re.compile(
    r"\b(remind me|set a reminder|don'?t let me forget|would you remind|can you remind|could you remind|will you remind)\b",
    re.IGNORECASE,
)


def _parse_structured_detail(detail: str) -> dict[str, Any]:
    """Best-effort structured detail for file actions.

    Prefers JSON ({"path": ..., "content": ...}); falls back to a
    "<path> :: <body>" convention so a small model that skips JSON still works.
    """
    text = (detail or "").strip()
    if not text:
        return {}
    if text.startswith("{"):
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    if "::" in text:
        path, _, body = text.partition("::")
        return {"path": path.strip(), "content": body.strip()}
    return {"path": text}


def reminder_request_supported(user_input: str) -> bool:
    """True when REMINDER's permission gate would allow the action to run.

    Routing uses this so a fuzzy (embedding/LLM) route to reminder_reply can't
    schedule an action that is guaranteed to fail its own gate.
    """
    if _EXPLICIT_REMINDER.search(user_input or ""):
        return True
    return bool(extract_commitments(user_input or "", time.time()))


_CALC_LEADING_PHRASE = re.compile(
    r"^\s*(?:what(?:'?s| is)|calculate|compute|solve|evaluate|=)\s*",
    re.IGNORECASE,
)
_CALC_PERCENT_OF = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s*of\s+", re.IGNORECASE
)
_CALC_WORD_OPERATORS = [
    (re.compile(r"\bmultiplied by\b", re.IGNORECASE), "*"),
    (re.compile(r"\bdivided by\b", re.IGNORECASE), "/"),
    (re.compile(r"\btimes\b", re.IGNORECASE), "*"),
    (re.compile(r"\bplus\b", re.IGNORECASE), "+"),
    (re.compile(r"\bminus\b", re.IGNORECASE), "-"),
]


def _normalize_expression(text: str) -> str:
    """Rewrite common natural-language arithmetic into the allowed grammar.

    Only produces pure-arithmetic strings; anything it cannot turn into
    arithmetic is left for ast.parse/the AST whitelist to reject cleanly.
    """
    value = (text or "").strip()
    if not value:
        return value
    value = value.rstrip("?").strip()
    value = _CALC_LEADING_PHRASE.sub("", value)
    # "15% of 200" / "15 percent of 200" -> "(15/100)*200"
    value = _CALC_PERCENT_OF.sub(lambda m: f"({m.group(1)}/100)*", value)
    for pattern, symbol in _CALC_WORD_OPERATORS:
        value = pattern.sub(symbol, value)
    return value.strip()


def calculate(expression: str) -> int | float:
    text = _normalize_expression(expression)
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
        files_client: Any | None = None,
        files_writer: Any | None = None,
        web_fetch_client: Any | None = None,
        file_parse_client: Any | None = None,
        approvals: Any | None = None,
        tier_for: Any | None = None,
    ) -> None:
        self.search_client = search_client or DEFAULT_SEARCH_CLIENT
        self.memory_client = memory_client
        self._files_client = files_client
        self._files_writer = files_writer
        self._web_fetch_client = web_fetch_client
        self._file_parse_client = file_parse_client
        # M3: the approvals store (T2 gate) and the tier classifier. tier_for is
        # injectable so the gate can be exercised without a shipped T2 action.
        self.approvals = approvals
        self._tier_for = tier_for or action_tier

    @property
    def files_client(self):
        # Lazy so tests and setups that never touch NOTE_READ skip config/root resolution.
        if self._files_client is None:
            from tools.localfiles import LocalFilesClient

            self._files_client = LocalFilesClient()
        return self._files_client

    @property
    def files_writer(self):
        if self._files_writer is None:
            from tools.localfiles import LocalFilesWriter

            self._files_writer = LocalFilesWriter()
        return self._files_writer

    @property
    def web_fetch_client(self):
        if self._web_fetch_client is None:
            from tools.webfetch import WebFetchClient

            self._web_fetch_client = WebFetchClient()
        return self._web_fetch_client

    @property
    def file_parse_client(self):
        if self._file_parse_client is None:
            from tools.fileparse import FileParseClient

            self._file_parse_client = FileParseClient()
        return self._file_parse_client

    def _require_memory(self) -> MemoryActionClient:
        if self.memory_client is None:
            raise RuntimeError("Memory actions are unavailable")
        return self.memory_client

    def _await_approval(
        self,
        action: Action,
        started_at: str,
        started: float,
        turn_id: str,
        session_id: str,
    ) -> ActionResult:
        summary = f"{action.action_type.value.replace('_', ' ').title()}: {action.detail}".strip()
        error = ""
        output: dict[str, Any]
        if self.approvals is None:
            # Fail closed: with nowhere to record the request, refuse rather
            # than execute an unreviewed T2 action.
            error = "This action needs your approval, but the approvals store is unavailable."
            output = {"status": "blocked", "summary": summary}
        else:
            approval = self.approvals.create_approval(
                action_type=action.action_type.value,
                summary=summary,
                payload={"detail": action.detail},
                turn_id=turn_id,
                session_id=session_id,
            )
            record = getattr(self.approvals, "record_action_audit", None)
            if callable(record):
                record(
                    action_type=action.action_type.value,
                    tier="T2",
                    detail=action.detail,
                    ok=True,
                    approval_id=approval["id"],
                    outcome="awaiting_approval",
                    turn_id=turn_id,
                )
            output = {
                "status": "awaiting_approval",
                "approval_id": approval["id"],
                "summary": summary,
                "note": (
                    "This action was NOT performed. It is waiting for the user's "
                    "explicit approval and will run only once they approve it."
                ),
            }
        return ActionResult(
            action_id=action.action_id,
            action_type=action.action_type,
            detail=action.detail,
            ok=not error,
            output=output,
            sources=[],
            error=error,
            started_at=started_at,
            completed_at=utc_now(),
            duration_ms=max(0, int((time.perf_counter() - started) * 1000)),
        )

    def run(
        self,
        action: Action,
        *,
        user_input: str,
        search_cache: dict[str, list[SearchResult]] | None = None,
        approved: bool = False,
        turn_id: str = "",
        session_id: str = "",
    ) -> ActionResult:
        started_at = utc_now()
        started = time.perf_counter()
        output: Any = None
        sources: list[SourceRef] = []
        error = ""

        # T2 gate: an outbound/irreversible action never executes on first sight.
        # It pauses as a pending approval and only runs when re-invoked with
        # approved=True by the approval flow.
        if not approved and self._tier_for(action.action_type) == "T2":
            return self._await_approval(
                action, started_at, started, turn_id, session_id
            )

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
                query = action.detail or user_input
                memory_context = self._require_memory().search_memory(query)
                pack = ContextPack.from_memory(memory_context)
                output = pack.to_dict()
                sources = list(pack.sources)
            elif action.action_type == ActionType.SUMMARIZE:
                memory = self._require_memory()
                scope = action.detail or user_input
                notes_getter = getattr(memory, "recent_notes", None)
                items = notes_getter(limit=20) if callable(notes_getter) else []
                lines = []
                for item in items:
                    lines.append(
                        f"[memory:item:{item.get('id')}] "
                        f"({item.get('item_type', 'note')}) {item.get('content', '')}"
                    )
                    sources.append(
                        SourceRef(
                            source_id=f"memory:item:{item.get('id')}",
                            kind="memory_item",
                            title=str(item.get("content", ""))[:80],
                            excerpt=str(item.get("content", "")),
                        )
                    )
                output = {
                    "scope": scope,
                    "material": lines,
                    "instruction": (
                        "Summarize ONLY the material above into what the user asked for "
                        "(summary, checklist, or bullets). Cite item IDs."
                        if lines
                        else "No stored notes matched. Say so plainly; do not invent notes."
                    ),
                }
            elif action.action_type == ActionType.REMINDER:
                memory = self._require_memory()
                now = time.time()
                if _EXPLICIT_REMINDER.search(user_input or ""):
                    parsed = parse_reminder(action.detail or user_input, now)
                    if parsed is None:
                        parsed = parse_reminder(user_input, now)
                else:
                    # Self-declared commitments ("I promised X by Friday") may
                    # be tracked too — memory capture would store them anyway.
                    commitments = extract_commitments(user_input or "", now)
                    parsed = commitments[0] if commitments else None
                    if parsed is None:
                        raise PermissionError(
                            "REMINDER requires an explicit user request to be reminded"
                        )
                if parsed is None or not parsed["content"]:
                    raise ValueError("Could not parse what to be reminded about")
                if parsed["due_at"] is None:
                    raise ValueError(
                        "A reminder needs an exact time, such as 'in 20 minutes', "
                        "'tomorrow morning', or 'at 2pm'. Nothing was stored."
                    )
                creator = getattr(memory, "create_reminder", None)
                if not callable(creator):
                    raise RuntimeError("Reminder storage is unavailable")
                item = creator(parsed["content"], due_at=parsed["due_at"])
                # upsert_item may match a locked (corrected/archived) item and
                # keep its existing content while returning the correct id, so
                # cite the stored content, not what we tried to write.
                stored_content = item.get("content") or parsed["content"]
                due_at = parsed["due_at"]
                output = {
                    "content": parsed["content"],
                    "due_at": due_at,
                    "due_at_human": (
                        time.strftime("%Y-%m-%d %H:%M", time.localtime(due_at))
                        if due_at
                        else "no specific time"
                    ),
                    "stored": True,
                }
                sources = [
                    SourceRef(
                        source_id=f"memory:item:{item['id']}",
                        kind="memory_item",
                        title=f"Reminder: {stored_content}",
                        excerpt=stored_content,
                    )
                ]
            elif action.action_type == ActionType.NOTE_READ:
                result = self.files_client.run(action.detail or user_input)
                output = result
                if not result.get("error"):
                    identity = result.get("path") or result.get("directory") or result.get("term", "")
                    sources = [
                        SourceRef(
                            source_id=stable_source_id("file", str(identity)),
                            kind="local_file",
                            title=str(identity) or "local files",
                            excerpt=str(result.get("content", ""))[:200],
                        )
                    ]
            elif action.action_type == ActionType.FILE_WRITE:
                spec = _parse_structured_detail(action.detail)
                path = str(spec.get("path") or "").strip()
                if not path:
                    raise ValueError("FILE_WRITE requires a target path")
                result = self.files_writer.write_file(path, str(spec.get("content", "")))
                output = result
                if result.get("error"):
                    raise ValueError(result["error"])
                sources = [
                    SourceRef(
                        source_id=stable_source_id("file", result["path"]),
                        kind="local_file",
                        title=result["path"],
                        excerpt=str(spec.get("content", ""))[:200],
                        metadata={"backup": result.get("backup"), "tier": "T1"},
                    )
                ]
            elif action.action_type == ActionType.FILE_EDIT:
                spec = _parse_structured_detail(action.detail)
                path = str(spec.get("path") or "").strip()
                if not path or "find" not in spec:
                    raise ValueError(
                        "FILE_EDIT requires a path, 'find' text, and 'replace' text"
                    )
                result = self.files_writer.edit_file(
                    path, str(spec.get("find", "")), str(spec.get("replace", ""))
                )
                output = result
                if result.get("error"):
                    raise ValueError(result["error"])
                sources = [
                    SourceRef(
                        source_id=stable_source_id("file", result["path"]),
                        kind="local_file",
                        title=result["path"],
                        excerpt=str(spec.get("replace", ""))[:200],
                        metadata={"backup": result.get("backup"), "tier": "T1"},
                    )
                ]
            elif action.action_type == ActionType.WEB_FETCH:
                result = self.web_fetch_client.fetch(action.detail or user_input)
                output = result
                if result.get("error"):
                    raise ValueError(result["error"])
                sources = [self.web_fetch_client.source_for(result)]
            elif action.action_type == ActionType.FILE_PARSE:
                result = self.file_parse_client.parse(action.detail or user_input)
                output = result
                if result.get("error"):
                    raise ValueError(result["error"])
                sources = [
                    SourceRef(
                        source_id=stable_source_id("file", result["path"]),
                        kind="local_file",
                        title=result["path"],
                        excerpt=str(result.get("content", ""))[:200],
                        metadata={"tier": "T0", "kind": result.get("kind", "")},
                    )
                ]
            elif action.action_type == ActionType.CLARIFY:
                question = (action.detail or "").strip()
                if not question:
                    raise ValueError("CLARIFY requires the question to ask")
                output = {"question": question}
            elif action.action_type == ActionType.RESEARCH:
                memory = self._require_memory()
                topic = (action.detail or user_input).strip()
                queuer = getattr(memory, "queue_research", None)
                if not callable(queuer):
                    raise RuntimeError("Background research is unavailable")
                job = queuer(topic)
                output = {
                    "status": "queued",
                    "topic": topic,
                    "job_id": job.get("job_id"),
                    "note": "Research runs in the background; results arrive as a proactive message.",
                }
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
